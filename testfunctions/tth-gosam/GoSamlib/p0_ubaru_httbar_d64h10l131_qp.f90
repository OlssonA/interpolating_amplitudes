module     p0_ubaru_httbar_d64h10l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d64h10l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1mu0 = 0
   integer, parameter :: ninjaidxt0mu0 = 1
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd64h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd64
      complex(ki), dimension (0:*), intent(inout) :: brack
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd64h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(34) :: acd64
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd64(1)=dotproduct(k2,ninjaE3)
      acd64(2)=dotproduct(ninjaE3,spvak2k1)
      acd64(3)=abb64(10)
      acd64(4)=dotproduct(ninjaE3,spvak2l3)
      acd64(5)=abb64(11)
      acd64(6)=dotproduct(k2,ninjaA)
      acd64(7)=dotproduct(ninjaA,spvak2k1)
      acd64(8)=abb64(13)
      acd64(9)=dotproduct(ninjaA,ninjaE3)
      acd64(10)=abb64(15)
      acd64(11)=dotproduct(ninjaA,spvak2l3)
      acd64(12)=abb64(9)
      acd64(13)=abb64(28)
      acd64(14)=dotproduct(ninjaE3,spval4k1)
      acd64(15)=abb64(17)
      acd64(16)=dotproduct(ninjaE3,spval4l5)
      acd64(17)=abb64(19)
      acd64(18)=dotproduct(ninjaE3,spval3k1)
      acd64(19)=abb64(20)
      acd64(20)=dotproduct(ninjaE3,spval3l5)
      acd64(21)=abb64(21)
      acd64(22)=dotproduct(ninjaE3,spvak2l5)
      acd64(23)=abb64(24)
      acd64(24)=acd64(3)*acd64(1)
      acd64(25)=acd64(5)*acd64(4)
      acd64(24)=acd64(24)+acd64(25)
      acd64(25)=acd64(2)*acd64(24)
      acd64(26)=acd64(6)*acd64(3)
      acd64(27)=acd64(11)*acd64(5)
      acd64(26)=acd64(12)+acd64(27)+acd64(26)
      acd64(26)=acd64(2)*acd64(26)
      acd64(24)=acd64(7)*acd64(24)
      acd64(27)=acd64(8)*acd64(1)
      acd64(28)=acd64(10)*acd64(9)
      acd64(29)=acd64(13)*acd64(4)
      acd64(30)=acd64(15)*acd64(14)
      acd64(31)=acd64(17)*acd64(16)
      acd64(32)=acd64(19)*acd64(18)
      acd64(33)=acd64(21)*acd64(20)
      acd64(34)=acd64(23)*acd64(22)
      acd64(24)=acd64(34)+acd64(33)+acd64(32)+acd64(31)+acd64(30)+acd64(29)+2.0&
      &_ki*acd64(28)+acd64(27)+acd64(24)+acd64(26)
      brack(ninjaidxt1mu0)=acd64(25)
      brack(ninjaidxt0mu0)=acd64(24)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d64h10_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd64h10_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-2))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d64h10l131_qp
