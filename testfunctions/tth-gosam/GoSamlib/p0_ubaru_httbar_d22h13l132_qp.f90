module     p0_ubaru_httbar_d22h13l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d22h13l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2x0mu0 = 0
   integer, parameter :: ninjaidxt1x0mu0 = 1
   integer, parameter :: ninjaidxt1x1mu0 = 2
   integer, parameter :: ninjaidxt0x0mu0 = 3
   integer, parameter :: ninjaidxt0x0mu2 = 4
   integer, parameter :: ninjaidxt0x1mu0 = 5
   integer, parameter :: ninjaidxt0x2mu0 = 6
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd22h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(30) :: acd22
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd22(1)=dotproduct(k2,ninjaE3)
      acd22(2)=abb22(13)
      acd22(3)=dotproduct(l3,ninjaE3)
      acd22(4)=abb22(17)
      acd22(5)=dotproduct(l5,ninjaE3)
      acd22(6)=abb22(35)
      acd22(7)=dotproduct(ninjaE3,spvak2l3)
      acd22(8)=abb22(12)
      acd22(9)=dotproduct(ninjaE3,spvak2l5)
      acd22(10)=abb22(14)
      acd22(11)=dotproduct(ninjaE3,spvak1l5)
      acd22(12)=abb22(16)
      acd22(13)=dotproduct(ninjaE3,spvak1l3)
      acd22(14)=abb22(18)
      acd22(15)=dotproduct(ninjaE3,spval5l3)
      acd22(16)=abb22(19)
      acd22(17)=dotproduct(ninjaE3,spval3l5)
      acd22(18)=abb22(21)
      acd22(19)=dotproduct(ninjaE3,spval3k2)
      acd22(20)=abb22(22)
      acd22(21)=acd22(2)*acd22(1)
      acd22(22)=acd22(4)*acd22(3)
      acd22(23)=acd22(6)*acd22(5)
      acd22(24)=acd22(8)*acd22(7)
      acd22(25)=acd22(10)*acd22(9)
      acd22(26)=acd22(12)*acd22(11)
      acd22(27)=acd22(14)*acd22(13)
      acd22(28)=acd22(16)*acd22(15)
      acd22(29)=acd22(18)*acd22(17)
      acd22(30)=acd22(20)*acd22(19)
      acd22(21)=acd22(30)+acd22(29)+acd22(28)+acd22(27)+acd22(26)+acd22(25)+acd&
      &22(24)+acd22(23)+acd22(21)+acd22(22)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd22(21)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd22h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(42) :: acd22
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd22(1)=dotproduct(k2,ninjaA1)
      acd22(2)=abb22(13)
      acd22(3)=dotproduct(l3,ninjaA1)
      acd22(4)=abb22(17)
      acd22(5)=dotproduct(l5,ninjaA1)
      acd22(6)=abb22(35)
      acd22(7)=dotproduct(ninjaA1,spvak2l3)
      acd22(8)=abb22(12)
      acd22(9)=dotproduct(ninjaA1,spvak2l5)
      acd22(10)=abb22(14)
      acd22(11)=dotproduct(ninjaA1,spvak1l5)
      acd22(12)=abb22(16)
      acd22(13)=dotproduct(ninjaA1,spvak1l3)
      acd22(14)=abb22(18)
      acd22(15)=dotproduct(ninjaA1,spval5l3)
      acd22(16)=abb22(19)
      acd22(17)=dotproduct(ninjaA1,spval3l5)
      acd22(18)=abb22(21)
      acd22(19)=dotproduct(ninjaA1,spval3k2)
      acd22(20)=abb22(22)
      acd22(21)=dotproduct(k2,ninjaA0)
      acd22(22)=dotproduct(l3,ninjaA0)
      acd22(23)=dotproduct(l5,ninjaA0)
      acd22(24)=dotproduct(ninjaA0,spvak2l3)
      acd22(25)=dotproduct(ninjaA0,spvak2l5)
      acd22(26)=dotproduct(ninjaA0,spvak1l5)
      acd22(27)=dotproduct(ninjaA0,spvak1l3)
      acd22(28)=dotproduct(ninjaA0,spval5l3)
      acd22(29)=dotproduct(ninjaA0,spval3l5)
      acd22(30)=dotproduct(ninjaA0,spval3k2)
      acd22(31)=abb22(15)
      acd22(32)=acd22(1)*acd22(2)
      acd22(33)=acd22(3)*acd22(4)
      acd22(34)=acd22(5)*acd22(6)
      acd22(35)=acd22(7)*acd22(8)
      acd22(36)=acd22(9)*acd22(10)
      acd22(37)=acd22(11)*acd22(12)
      acd22(38)=acd22(13)*acd22(14)
      acd22(39)=acd22(15)*acd22(16)
      acd22(40)=acd22(17)*acd22(18)
      acd22(41)=acd22(19)*acd22(20)
      acd22(32)=acd22(41)+acd22(40)+acd22(39)+acd22(38)+acd22(37)+acd22(36)+acd&
      &22(35)+acd22(34)+acd22(32)+acd22(33)
      acd22(33)=acd22(21)*acd22(2)
      acd22(34)=acd22(22)*acd22(4)
      acd22(35)=acd22(23)*acd22(6)
      acd22(36)=acd22(24)*acd22(8)
      acd22(37)=acd22(25)*acd22(10)
      acd22(38)=acd22(26)*acd22(12)
      acd22(39)=acd22(27)*acd22(14)
      acd22(40)=acd22(28)*acd22(16)
      acd22(41)=acd22(29)*acd22(18)
      acd22(42)=acd22(30)*acd22(20)
      acd22(33)=acd22(31)+acd22(42)+acd22(41)+acd22(40)+acd22(39)+acd22(38)+acd&
      &22(37)+acd22(36)+acd22(35)+acd22(33)+acd22(34)
      brack(ninjaidxt0x0mu0)=acd22(33)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd22(32)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d22h13_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd22h13_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA0(1:4) = - a0(0:3)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p0_ubaru_httbar_d22h13l132_qp
