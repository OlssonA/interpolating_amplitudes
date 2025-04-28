module     p2_gg_httbar_d46h8l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d46h8l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd46h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(97) :: acd46
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd46(1)=dotproduct(k1,ninjaA)
      acd46(2)=dotproduct(k2,ninjaE3)
      acd46(3)=abb46(19)
      acd46(4)=dotproduct(ninjaE3,spval4l5)
      acd46(5)=abb46(9)
      acd46(6)=dotproduct(ninjaE3,spvak2l3)
      acd46(7)=abb46(16)
      acd46(8)=dotproduct(ninjaE3,spval3k2)
      acd46(9)=abb46(20)
      acd46(10)=dotproduct(ninjaE3,spval4l3)
      acd46(11)=abb46(33)
      acd46(12)=dotproduct(ninjaE3,spval3l5)
      acd46(13)=abb46(35)
      acd46(14)=dotproduct(k1,ninjaE3)
      acd46(15)=dotproduct(k2,ninjaA)
      acd46(16)=dotproduct(ninjaA,spval4l5)
      acd46(17)=dotproduct(ninjaA,spvak2l3)
      acd46(18)=dotproduct(ninjaA,spval3k2)
      acd46(19)=dotproduct(ninjaA,spval4l3)
      acd46(20)=dotproduct(ninjaA,spval3l5)
      acd46(21)=abb46(22)
      acd46(22)=abb46(13)
      acd46(23)=abb46(34)
      acd46(24)=abb46(43)
      acd46(25)=abb46(25)
      acd46(26)=dotproduct(ninjaA,ninjaE3)
      acd46(27)=abb46(23)
      acd46(28)=dotproduct(ninjaA,spval4k1)
      acd46(29)=dotproduct(ninjaE3,spvak1l5)
      acd46(30)=abb46(12)
      acd46(31)=dotproduct(ninjaE3,spvak1l3)
      acd46(32)=abb46(44)
      acd46(33)=dotproduct(ninjaA,spvak1l5)
      acd46(34)=dotproduct(ninjaE3,spval4k1)
      acd46(35)=dotproduct(ninjaE3,spval3k1)
      acd46(36)=abb46(24)
      acd46(37)=dotproduct(ninjaA,spvak2l5)
      acd46(38)=dotproduct(ninjaE3,spval4k2)
      acd46(39)=dotproduct(ninjaA,spval4k2)
      acd46(40)=dotproduct(ninjaE3,spvak2l5)
      acd46(41)=dotproduct(ninjaA,spvak1k2)
      acd46(42)=dotproduct(ninjaE3,spvak2k1)
      acd46(43)=abb46(14)
      acd46(44)=abb46(30)
      acd46(45)=dotproduct(ninjaA,spvak2k1)
      acd46(46)=dotproduct(ninjaE3,spvak1k2)
      acd46(47)=abb46(15)
      acd46(48)=dotproduct(ninjaA,spvak1l3)
      acd46(49)=dotproduct(ninjaA,spval3k1)
      acd46(50)=abb46(10)
      acd46(51)=abb46(11)
      acd46(52)=abb46(45)
      acd46(53)=abb46(27)
      acd46(54)=abb46(31)
      acd46(55)=abb46(32)
      acd46(56)=abb46(29)
      acd46(57)=abb46(26)
      acd46(58)=abb46(38)
      acd46(59)=abb46(39)
      acd46(60)=-acd46(16)*acd46(5)
      acd46(61)=acd46(19)*acd46(11)
      acd46(62)=acd46(20)*acd46(13)
      acd46(60)=acd46(62)+acd46(61)+acd46(60)
      acd46(61)=acd46(14)-acd46(2)
      acd46(60)=acd46(61)*acd46(60)
      acd46(62)=acd46(4)*acd46(5)
      acd46(63)=acd46(10)*acd46(11)
      acd46(64)=acd46(12)*acd46(13)
      acd46(62)=-acd46(64)+acd46(62)-acd46(63)
      acd46(63)=acd46(1)-acd46(15)
      acd46(63)=-acd46(63)*acd46(62)
      acd46(64)=acd46(9)*acd46(8)
      acd46(65)=acd46(7)*acd46(6)
      acd46(65)=acd46(64)+acd46(65)
      acd46(65)=acd46(1)*acd46(65)
      acd46(66)=acd46(15)*acd46(14)
      acd46(67)=acd46(1)*acd46(2)
      acd46(66)=acd46(66)+acd46(67)
      acd46(66)=acd46(3)*acd46(66)
      acd46(67)=acd46(38)*acd46(32)
      acd46(68)=acd46(7)*acd46(14)
      acd46(67)=acd46(67)+acd46(68)
      acd46(68)=acd46(17)*acd46(67)
      acd46(69)=acd46(9)*acd46(14)
      acd46(70)=acd46(40)*acd46(36)
      acd46(69)=acd46(70)+acd46(69)
      acd46(69)=acd46(18)*acd46(69)
      acd46(70)=acd46(15)*acd46(6)
      acd46(71)=acd46(17)*acd46(2)
      acd46(70)=acd46(70)+acd46(71)
      acd46(70)=acd46(23)*acd46(70)
      acd46(71)=acd46(15)*acd46(8)
      acd46(72)=acd46(18)*acd46(2)
      acd46(71)=acd46(71)+acd46(72)
      acd46(71)=acd46(24)*acd46(71)
      acd46(72)=acd46(29)*acd46(30)
      acd46(73)=acd46(31)*acd46(32)
      acd46(72)=acd46(72)+acd46(73)
      acd46(73)=-acd46(28)*acd46(72)
      acd46(74)=-acd46(34)*acd46(30)
      acd46(75)=-acd46(35)*acd46(36)
      acd46(74)=acd46(74)+acd46(75)
      acd46(74)=acd46(33)*acd46(74)
      acd46(75)=acd46(36)*acd46(8)
      acd46(76)=acd46(38)*acd46(30)
      acd46(75)=acd46(75)+acd46(76)
      acd46(76)=acd46(37)*acd46(75)
      acd46(77)=acd46(32)*acd46(6)
      acd46(78)=acd46(40)*acd46(30)
      acd46(77)=acd46(77)+acd46(78)
      acd46(77)=acd46(39)*acd46(77)
      acd46(78)=acd46(43)*acd46(42)
      acd46(79)=acd46(44)*acd46(35)
      acd46(78)=acd46(78)+acd46(79)
      acd46(78)=acd46(41)*acd46(78)
      acd46(79)=acd46(43)*acd46(46)
      acd46(80)=acd46(47)*acd46(31)
      acd46(80)=acd46(79)+acd46(80)
      acd46(80)=acd46(45)*acd46(80)
      acd46(81)=acd46(47)*acd46(42)
      acd46(82)=-acd46(34)*acd46(32)
      acd46(82)=acd46(82)+acd46(81)
      acd46(82)=acd46(48)*acd46(82)
      acd46(83)=acd46(29)*acd46(36)
      acd46(84)=acd46(44)*acd46(46)
      acd46(83)=acd46(83)-acd46(84)
      acd46(84)=-acd46(49)*acd46(83)
      acd46(85)=acd46(22)*acd46(15)
      acd46(85)=acd46(25)+2.0_ki*acd46(85)
      acd46(85)=acd46(2)*acd46(85)
      acd46(86)=acd46(21)*acd46(14)
      acd46(87)=acd46(27)*acd46(26)
      acd46(88)=acd46(50)*acd46(34)
      acd46(89)=acd46(51)*acd46(29)
      acd46(90)=acd46(52)*acd46(40)
      acd46(91)=acd46(53)*acd46(38)
      acd46(92)=acd46(54)*acd46(46)
      acd46(93)=acd46(55)*acd46(42)
      acd46(94)=acd46(56)*acd46(31)
      acd46(95)=acd46(57)*acd46(6)
      acd46(96)=acd46(58)*acd46(8)
      acd46(97)=acd46(59)*acd46(35)
      acd46(60)=acd46(97)+acd46(96)+acd46(95)+acd46(94)+acd46(93)+acd46(92)+acd&
      &46(91)+acd46(90)+acd46(89)+acd46(88)+2.0_ki*acd46(87)+acd46(86)+acd46(84&
      &)+acd46(82)+acd46(80)+acd46(78)+acd46(77)+acd46(76)+acd46(74)+acd46(73)+&
      &acd46(71)+acd46(70)+acd46(69)+acd46(68)+acd46(66)+acd46(63)+acd46(60)+ac&
      &d46(65)+acd46(85)
      acd46(61)=-acd46(61)*acd46(62)
      acd46(62)=acd46(3)*acd46(2)
      acd46(62)=acd46(64)+acd46(62)
      acd46(62)=acd46(14)*acd46(62)
      acd46(63)=-acd46(35)*acd46(83)
      acd46(64)=acd46(23)*acd46(2)
      acd46(64)=acd46(64)+acd46(67)
      acd46(64)=acd46(6)*acd46(64)
      acd46(65)=-acd46(34)*acd46(72)
      acd46(66)=acd46(40)*acd46(75)
      acd46(67)=acd46(24)*acd46(8)*acd46(2)
      acd46(68)=acd46(42)*acd46(79)
      acd46(69)=acd46(31)*acd46(81)
      acd46(70)=acd46(22)*acd46(2)**2
      acd46(61)=acd46(70)+acd46(69)+acd46(68)+acd46(67)+acd46(66)+acd46(65)+acd&
      &46(61)+acd46(64)+acd46(63)+acd46(62)
      brack(ninjaidxt3mu0)=acd46(61)
      brack(ninjaidxt2mu0)=acd46(60)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d46h8_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd46h8
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k4-k5
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d46h8l131
