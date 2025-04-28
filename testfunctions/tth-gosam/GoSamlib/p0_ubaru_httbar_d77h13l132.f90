module     p0_ubaru_httbar_d77h13l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d77h13l132.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond_t, d => metric_tensor
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
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd77h13
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(26) :: acd77
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd77(1)=dotproduct(k2,ninjaE3)
      acd77(2)=dotproduct(ninjaE3,spvak1l4)
      acd77(3)=abb77(23)
      acd77(4)=dotproduct(ninjaE3,spvak1l5)
      acd77(5)=abb77(32)
      acd77(6)=dotproduct(ninjaE3,spvak1l3)
      acd77(7)=abb77(38)
      acd77(8)=dotproduct(ninjaE3,spvak1k2)
      acd77(9)=dotproduct(ninjaE3,spvak2l3)
      acd77(10)=abb77(12)
      acd77(11)=dotproduct(ninjaE3,spvak2l4)
      acd77(12)=abb77(13)
      acd77(13)=dotproduct(ninjaE3,spvak2l5)
      acd77(14)=abb77(14)
      acd77(15)=dotproduct(ninjaE3,spval3l5)
      acd77(16)=abb77(20)
      acd77(17)=dotproduct(ninjaE3,spval3l4)
      acd77(18)=abb77(24)
      acd77(19)=dotproduct(ninjaE3,spval3k2)
      acd77(20)=abb77(16)
      acd77(21)=abb77(41)
      acd77(22)=acd77(10)*acd77(9)
      acd77(23)=acd77(12)*acd77(11)
      acd77(24)=acd77(14)*acd77(13)
      acd77(25)=acd77(16)*acd77(15)
      acd77(26)=acd77(18)*acd77(17)
      acd77(22)=acd77(26)+acd77(25)+acd77(24)+acd77(23)+acd77(22)
      acd77(22)=acd77(8)*acd77(22)
      acd77(23)=acd77(3)*acd77(2)
      acd77(24)=acd77(5)*acd77(4)
      acd77(25)=acd77(7)*acd77(6)
      acd77(23)=acd77(25)+acd77(23)+acd77(24)
      acd77(23)=acd77(1)*acd77(23)
      acd77(24)=acd77(20)*acd77(2)
      acd77(25)=acd77(21)*acd77(4)
      acd77(24)=acd77(25)+acd77(24)
      acd77(24)=acd77(19)*acd77(24)
      acd77(22)=acd77(22)+acd77(23)+acd77(24)
      brack(ninjaidxt1x0mu0)=acd77(22)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd77h13
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(73) :: acd77
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd77(1)=dotproduct(k2,ninjaA1)
      acd77(2)=dotproduct(ninjaE3,spvak1l4)
      acd77(3)=abb77(23)
      acd77(4)=dotproduct(ninjaE3,spvak1l5)
      acd77(5)=abb77(32)
      acd77(6)=dotproduct(ninjaE3,spvak1l3)
      acd77(7)=abb77(38)
      acd77(8)=dotproduct(k2,ninjaE3)
      acd77(9)=dotproduct(ninjaA1,spvak1l4)
      acd77(10)=dotproduct(ninjaA1,spvak1l5)
      acd77(11)=dotproduct(ninjaA1,spvak1l3)
      acd77(12)=dotproduct(ninjaA1,spvak1k2)
      acd77(13)=dotproduct(ninjaE3,spvak2l3)
      acd77(14)=abb77(12)
      acd77(15)=dotproduct(ninjaE3,spvak2l4)
      acd77(16)=abb77(13)
      acd77(17)=dotproduct(ninjaE3,spvak2l5)
      acd77(18)=abb77(14)
      acd77(19)=dotproduct(ninjaE3,spval3l5)
      acd77(20)=abb77(20)
      acd77(21)=dotproduct(ninjaE3,spval3l4)
      acd77(22)=abb77(24)
      acd77(23)=dotproduct(ninjaA1,spvak2l3)
      acd77(24)=dotproduct(ninjaE3,spvak1k2)
      acd77(25)=dotproduct(ninjaA1,spvak2l4)
      acd77(26)=dotproduct(ninjaA1,spvak2l5)
      acd77(27)=dotproduct(ninjaE3,spval3k2)
      acd77(28)=abb77(16)
      acd77(29)=dotproduct(ninjaA1,spval3k2)
      acd77(30)=abb77(41)
      acd77(31)=dotproduct(ninjaA1,spval3l5)
      acd77(32)=dotproduct(ninjaA1,spval3l4)
      acd77(33)=dotproduct(k2,ninjaA0)
      acd77(34)=dotproduct(ninjaA0,spvak1l4)
      acd77(35)=dotproduct(ninjaA0,spvak1l5)
      acd77(36)=dotproduct(ninjaA0,spvak1l3)
      acd77(37)=abb77(15)
      acd77(38)=dotproduct(ninjaA0,ninjaE3)
      acd77(39)=abb77(11)
      acd77(40)=dotproduct(ninjaA0,spvak1k2)
      acd77(41)=dotproduct(ninjaA0,spvak2l3)
      acd77(42)=dotproduct(ninjaA0,spvak2l4)
      acd77(43)=dotproduct(ninjaA0,spvak2l5)
      acd77(44)=dotproduct(ninjaA0,spval3k2)
      acd77(45)=dotproduct(ninjaA0,spval3l5)
      acd77(46)=dotproduct(ninjaA0,spval3l4)
      acd77(47)=abb77(10)
      acd77(48)=abb77(17)
      acd77(49)=abb77(26)
      acd77(50)=abb77(30)
      acd77(51)=abb77(33)
      acd77(52)=acd77(13)*acd77(14)
      acd77(53)=acd77(15)*acd77(16)
      acd77(54)=acd77(17)*acd77(18)
      acd77(55)=acd77(19)*acd77(20)
      acd77(56)=acd77(21)*acd77(22)
      acd77(52)=acd77(56)+acd77(52)+acd77(53)+acd77(54)+acd77(55)
      acd77(53)=acd77(12)*acd77(52)
      acd77(54)=acd77(3)*acd77(2)
      acd77(55)=acd77(5)*acd77(4)
      acd77(56)=acd77(6)*acd77(7)
      acd77(54)=acd77(56)+acd77(54)+acd77(55)
      acd77(55)=acd77(1)*acd77(54)
      acd77(56)=acd77(3)*acd77(8)
      acd77(57)=acd77(28)*acd77(27)
      acd77(56)=acd77(56)+acd77(57)
      acd77(57)=acd77(9)*acd77(56)
      acd77(58)=acd77(5)*acd77(8)
      acd77(59)=acd77(30)*acd77(27)
      acd77(58)=acd77(58)+acd77(59)
      acd77(59)=acd77(10)*acd77(58)
      acd77(60)=acd77(28)*acd77(2)
      acd77(61)=acd77(30)*acd77(4)
      acd77(60)=acd77(60)+acd77(61)
      acd77(61)=acd77(29)*acd77(60)
      acd77(62)=acd77(7)*acd77(8)
      acd77(63)=acd77(11)*acd77(62)
      acd77(64)=acd77(14)*acd77(24)
      acd77(65)=acd77(23)*acd77(64)
      acd77(66)=acd77(16)*acd77(24)
      acd77(67)=acd77(25)*acd77(66)
      acd77(68)=acd77(18)*acd77(24)
      acd77(69)=acd77(26)*acd77(68)
      acd77(70)=acd77(20)*acd77(24)
      acd77(71)=acd77(31)*acd77(70)
      acd77(72)=acd77(22)*acd77(24)
      acd77(73)=acd77(32)*acd77(72)
      acd77(53)=acd77(73)+acd77(71)+acd77(69)+acd77(67)+acd77(65)+acd77(63)+acd&
      &77(61)+acd77(59)+acd77(57)+acd77(53)+acd77(55)
      acd77(52)=acd77(40)*acd77(52)
      acd77(54)=acd77(33)*acd77(54)
      acd77(55)=acd77(34)*acd77(56)
      acd77(56)=acd77(35)*acd77(58)
      acd77(57)=acd77(44)*acd77(60)
      acd77(58)=acd77(36)*acd77(62)
      acd77(59)=acd77(37)*acd77(8)
      acd77(60)=acd77(39)*acd77(38)
      acd77(61)=acd77(41)*acd77(64)
      acd77(62)=acd77(42)*acd77(66)
      acd77(63)=acd77(43)*acd77(68)
      acd77(64)=acd77(45)*acd77(70)
      acd77(65)=acd77(46)*acd77(72)
      acd77(66)=acd77(47)*acd77(24)
      acd77(67)=acd77(48)*acd77(2)
      acd77(68)=acd77(49)*acd77(27)
      acd77(69)=acd77(50)*acd77(4)
      acd77(70)=acd77(51)*acd77(6)
      acd77(52)=acd77(70)+acd77(69)+acd77(68)+acd77(67)+acd77(66)+acd77(65)+acd&
      &77(64)+acd77(63)+acd77(62)+acd77(61)+2.0_ki*acd77(60)+acd77(59)+acd77(58&
      &)+acd77(57)+acd77(56)+acd77(55)+acd77(54)+acd77(52)
      brack(ninjaidxt0x0mu0)=acd77(52)
      brack(ninjaidxt0x1mu0)=acd77(53)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d77h13_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd77h13
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2
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
end module     p0_ubaru_httbar_d77h13l132
