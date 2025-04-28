module     p0_ubaru_httbar_d13h14l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity14d13h14l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd13h14_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(7) :: acd13
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd13(1)=dotproduct(ninjaE3,spvak2k1)
      acd13(2)=dotproduct(ninjaE3,spvak2l4)
      acd13(3)=abb13(7)
      acd13(4)=dotproduct(ninjaE3,spvak2l5)
      acd13(5)=abb13(9)
      acd13(6)=acd13(3)*acd13(2)
      acd13(7)=acd13(5)*acd13(4)
      acd13(6)=acd13(6)+acd13(7)
      acd13(6)=acd13(1)*acd13(6)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd13(6)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd13h14_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(26) :: acd13
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd13(1)=dotproduct(ninjaE3,spvak2k1)
      acd13(2)=dotproduct(ninjaE4,spvak2l4)
      acd13(3)=abb13(7)
      acd13(4)=dotproduct(ninjaE4,spvak2l5)
      acd13(5)=abb13(9)
      acd13(6)=dotproduct(ninjaE3,spvak2l4)
      acd13(7)=dotproduct(ninjaE4,spvak2k1)
      acd13(8)=dotproduct(ninjaE3,spvak2l5)
      acd13(9)=dotproduct(ninjaA,spvak2k1)
      acd13(10)=dotproduct(ninjaA,spvak2l4)
      acd13(11)=dotproduct(ninjaA,spvak2l5)
      acd13(12)=abb13(8)
      acd13(13)=abb13(12)
      acd13(14)=abb13(13)
      acd13(15)=dotproduct(ninjaE3,spvak2l3)
      acd13(16)=abb13(10)
      acd13(17)=dotproduct(ninjaA,spvak2l3)
      acd13(18)=abb13(11)
      acd13(19)=acd13(6)*acd13(3)
      acd13(20)=acd13(8)*acd13(5)
      acd13(19)=acd13(19)+acd13(20)
      acd13(20)=acd13(19)*acd13(7)
      acd13(21)=acd13(5)*acd13(1)
      acd13(22)=acd13(21)*acd13(4)
      acd13(23)=acd13(3)*acd13(1)
      acd13(24)=acd13(23)*acd13(2)
      acd13(20)=acd13(20)+acd13(22)+acd13(24)
      acd13(19)=acd13(9)*acd13(19)
      acd13(22)=acd13(10)*acd13(23)
      acd13(21)=acd13(11)*acd13(21)
      acd13(23)=acd13(12)*acd13(1)
      acd13(24)=acd13(13)*acd13(6)
      acd13(25)=acd13(14)*acd13(8)
      acd13(26)=acd13(15)*acd13(16)
      acd13(19)=acd13(26)+acd13(25)+acd13(24)+acd13(23)+acd13(21)+acd13(22)+acd&
      &13(19)
      acd13(21)=ninjaP*acd13(20)
      acd13(22)=acd13(10)*acd13(3)
      acd13(23)=acd13(11)*acd13(5)
      acd13(22)=acd13(12)+acd13(23)+acd13(22)
      acd13(22)=acd13(9)*acd13(22)
      acd13(23)=acd13(13)*acd13(10)
      acd13(24)=acd13(14)*acd13(11)
      acd13(25)=acd13(17)*acd13(16)
      acd13(21)=acd13(18)+acd13(25)+acd13(24)+acd13(23)+acd13(21)+acd13(22)
      brack(ninjaidxt1mu0)=acd13(19)
      brack(ninjaidxt1mu2)=0.0_ki
      brack(ninjaidxt0mu0)=acd13(21)
      brack(ninjaidxt0mu2)=acd13(20)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d13h14_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd13h14_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k4
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d13h14l131_qp
