module     p0_ubaru_httbar_d71h5l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity5d71h5l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd71h5_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd71
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd71h5_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(42) :: acd71
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd71(1)=abb71(16)
      acd71(2)=dotproduct(k2,ninjaE3)
      acd71(3)=abb71(13)
      acd71(4)=dotproduct(l3,ninjaE3)
      acd71(5)=abb71(15)
      acd71(6)=dotproduct(l5,ninjaE3)
      acd71(7)=abb71(21)
      acd71(8)=dotproduct(ninjaA,ninjaE3)
      acd71(9)=dotproduct(ninjaE3,spval5k2)
      acd71(10)=abb71(10)
      acd71(11)=dotproduct(ninjaE3,spval3k2)
      acd71(12)=abb71(11)
      acd71(13)=dotproduct(ninjaE3,spvak1k2)
      acd71(14)=abb71(12)
      acd71(15)=dotproduct(ninjaE3,spvak1l3)
      acd71(16)=abb71(14)
      acd71(17)=dotproduct(ninjaE3,spval3l5)
      acd71(18)=abb71(20)
      acd71(19)=dotproduct(ninjaE3,spval5l3)
      acd71(20)=abb71(24)
      acd71(21)=dotproduct(k2,ninjaA)
      acd71(22)=dotproduct(l3,ninjaA)
      acd71(23)=dotproduct(l5,ninjaA)
      acd71(24)=dotproduct(ninjaA,ninjaA)
      acd71(25)=dotproduct(ninjaA,spval5k2)
      acd71(26)=dotproduct(ninjaA,spval3k2)
      acd71(27)=dotproduct(ninjaA,spvak1k2)
      acd71(28)=dotproduct(ninjaA,spvak1l3)
      acd71(29)=dotproduct(ninjaA,spval3l5)
      acd71(30)=dotproduct(ninjaA,spval5l3)
      acd71(31)=abb71(18)
      acd71(32)=acd71(2)*acd71(3)
      acd71(33)=acd71(4)*acd71(5)
      acd71(34)=acd71(6)*acd71(7)
      acd71(35)=acd71(8)*acd71(1)
      acd71(36)=acd71(9)*acd71(10)
      acd71(37)=acd71(11)*acd71(12)
      acd71(38)=acd71(13)*acd71(14)
      acd71(39)=acd71(15)*acd71(16)
      acd71(40)=acd71(17)*acd71(18)
      acd71(41)=acd71(19)*acd71(20)
      acd71(32)=acd71(41)+acd71(40)+acd71(39)+acd71(38)+acd71(37)+acd71(36)+2.0&
      &_ki*acd71(35)+acd71(34)+acd71(32)+acd71(33)
      acd71(33)=ninjaP+acd71(24)
      acd71(33)=acd71(1)*acd71(33)
      acd71(34)=acd71(21)*acd71(3)
      acd71(35)=acd71(22)*acd71(5)
      acd71(36)=acd71(23)*acd71(7)
      acd71(37)=acd71(25)*acd71(10)
      acd71(38)=acd71(26)*acd71(12)
      acd71(39)=acd71(27)*acd71(14)
      acd71(40)=acd71(28)*acd71(16)
      acd71(41)=acd71(29)*acd71(18)
      acd71(42)=acd71(30)*acd71(20)
      acd71(33)=acd71(31)+acd71(42)+acd71(41)+acd71(40)+acd71(39)+acd71(38)+acd&
      &71(37)+acd71(36)+acd71(34)+acd71(35)+acd71(33)
      brack(ninjaidxt1mu0)=acd71(32)
      brack(ninjaidxt0mu0)=acd71(33)
      brack(ninjaidxt0mu2)=acd71(1)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d71h5_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd71h5_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k5
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d71h5l131_qp
