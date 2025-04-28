module     p0_ubaru_httbar_d3h13l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d3h13l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1x0mu0 = 0
   integer, parameter :: ninjaidxt0x0mu0 = 1
   integer, parameter :: ninjaidxt0x1mu0 = 2
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd3h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(12) :: acd3
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd3(1)=dotproduct(k2,ninjaE3)
      acd3(2)=dotproduct(ninjaE3,spvak1l4)
      acd3(3)=abb3(12)
      acd3(4)=dotproduct(ninjaE3,spvak1l5)
      acd3(5)=abb3(13)
      acd3(6)=dotproduct(ninjaE3,spvak1l3)
      acd3(7)=abb3(15)
      acd3(8)=dotproduct(ninjaE3,spval3k2)
      acd3(9)=abb3(14)
      acd3(10)=acd3(3)*acd3(2)
      acd3(11)=acd3(5)*acd3(4)
      acd3(12)=acd3(7)*acd3(6)
      acd3(10)=acd3(12)+acd3(10)+acd3(11)
      acd3(10)=acd3(1)*acd3(10)
      acd3(11)=acd3(9)*acd3(8)*acd3(2)
      acd3(10)=acd3(11)+acd3(10)
      brack(ninjaidxt1x0mu0)=acd3(10)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd3h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(35) :: acd3
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd3(1)=dotproduct(k2,ninjaA1)
      acd3(2)=dotproduct(ninjaE3,spvak1l5)
      acd3(3)=abb3(13)
      acd3(4)=dotproduct(ninjaE3,spvak1l4)
      acd3(5)=abb3(12)
      acd3(6)=dotproduct(ninjaE3,spvak1l3)
      acd3(7)=abb3(15)
      acd3(8)=dotproduct(k2,ninjaE3)
      acd3(9)=dotproduct(ninjaA1,spvak1l5)
      acd3(10)=dotproduct(ninjaA1,spvak1l4)
      acd3(11)=dotproduct(ninjaA1,spvak1l3)
      acd3(12)=dotproduct(ninjaE3,spval3k2)
      acd3(13)=abb3(14)
      acd3(14)=dotproduct(ninjaA1,spval3k2)
      acd3(15)=dotproduct(k2,ninjaA0)
      acd3(16)=dotproduct(ninjaA0,spvak1l5)
      acd3(17)=dotproduct(ninjaA0,spvak1l4)
      acd3(18)=dotproduct(ninjaA0,spvak1l3)
      acd3(19)=abb3(22)
      acd3(20)=dotproduct(ninjaA0,spval3k2)
      acd3(21)=abb3(9)
      acd3(22)=abb3(10)
      acd3(23)=abb3(11)
      acd3(24)=abb3(23)
      acd3(25)=acd3(5)*acd3(4)
      acd3(26)=acd3(2)*acd3(3)
      acd3(27)=acd3(6)*acd3(7)
      acd3(25)=acd3(27)+acd3(25)+acd3(26)
      acd3(26)=acd3(1)*acd3(25)
      acd3(27)=acd3(5)*acd3(8)
      acd3(28)=acd3(12)*acd3(13)
      acd3(27)=acd3(27)+acd3(28)
      acd3(28)=acd3(10)*acd3(27)
      acd3(29)=acd3(3)*acd3(8)
      acd3(30)=acd3(9)*acd3(29)
      acd3(31)=acd3(7)*acd3(8)
      acd3(32)=acd3(11)*acd3(31)
      acd3(33)=acd3(13)*acd3(4)
      acd3(34)=acd3(14)*acd3(33)
      acd3(26)=acd3(34)+acd3(32)+acd3(30)+acd3(28)+acd3(26)
      acd3(25)=acd3(15)*acd3(25)
      acd3(27)=acd3(17)*acd3(27)
      acd3(28)=acd3(16)*acd3(29)
      acd3(29)=acd3(18)*acd3(31)
      acd3(30)=acd3(19)*acd3(8)
      acd3(31)=acd3(20)*acd3(33)
      acd3(32)=acd3(21)*acd3(2)
      acd3(33)=acd3(22)*acd3(4)
      acd3(34)=acd3(23)*acd3(6)
      acd3(35)=acd3(24)*acd3(12)
      acd3(25)=acd3(35)+acd3(34)+acd3(33)+acd3(32)+acd3(31)+acd3(30)+acd3(29)+a&
      &cd3(28)+acd3(25)+acd3(27)
      brack(ninjaidxt0x0mu0)=acd3(25)
      brack(ninjaidxt0x1mu0)=acd3(26)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d3h13_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd3h13_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k4
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p0_ubaru_httbar_d3h13l132_qp
