module     p2_gg_httbar_d260h4l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d260h4l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd260h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd260
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd260h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(128) :: acd260
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd260(1)=dotproduct(e2,ninjaE3)
      acd260(2)=dotproduct(ninjaE3,spvak2e1)
      acd260(3)=abb260(34)
      acd260(4)=dotproduct(ninjaE3,spvae1k2)
      acd260(5)=abb260(45)
      acd260(6)=dotproduct(ninjaE3,spval5e1)
      acd260(7)=abb260(96)
      acd260(8)=dotproduct(ninjaE3,spvae1l4)
      acd260(9)=abb260(51)
      acd260(10)=dotproduct(ninjaE3,spvae1e2)
      acd260(11)=abb260(71)
      acd260(12)=dotproduct(ninjaE3,spvae2e1)
      acd260(13)=abb260(86)
      acd260(14)=dotproduct(k2,ninjaE3)
      acd260(15)=abb260(56)
      acd260(16)=dotproduct(ninjaA,ninjaE3)
      acd260(17)=dotproduct(ninjaE3,spval3e1)
      acd260(18)=abb260(79)
      acd260(19)=abb260(142)
      acd260(20)=abb260(10)
      acd260(21)=dotproduct(ninjaE3,spvae1l3)
      acd260(22)=abb260(32)
      acd260(23)=abb260(135)
      acd260(24)=abb260(111)
      acd260(25)=dotproduct(ninjaE3,spval5l3)
      acd260(26)=dotproduct(ninjaE3,spval5k2)
      acd260(27)=abb260(27)
      acd260(28)=dotproduct(ninjaE3,spval3k2)
      acd260(29)=dotproduct(k2,ninjaA)
      acd260(30)=dotproduct(ninjaA,spvae1k2)
      acd260(31)=dotproduct(ninjaA,spvae2e1)
      acd260(32)=abb260(29)
      acd260(33)=abb260(37)
      acd260(34)=dotproduct(e2,ninjaA)
      acd260(35)=dotproduct(ninjaA,ninjaA)
      acd260(36)=dotproduct(ninjaA,spvak2e1)
      acd260(37)=dotproduct(ninjaA,spval5e1)
      acd260(38)=dotproduct(ninjaA,spvae1l4)
      acd260(39)=abb260(52)
      acd260(40)=dotproduct(ninjaA,spval3e1)
      acd260(41)=dotproduct(ninjaA,spvae1l3)
      acd260(42)=abb260(50)
      acd260(43)=abb260(20)
      acd260(44)=abb260(25)
      acd260(45)=abb260(59)
      acd260(46)=abb260(75)
      acd260(47)=abb260(46)
      acd260(48)=abb260(70)
      acd260(49)=dotproduct(ninjaA,spvae1e2)
      acd260(50)=abb260(58)
      acd260(51)=abb260(73)
      acd260(52)=abb260(24)
      acd260(53)=abb260(26)
      acd260(54)=abb260(61)
      acd260(55)=abb260(19)
      acd260(56)=abb260(94)
      acd260(57)=dotproduct(ninjaE3,spvak1l4)
      acd260(58)=abb260(72)
      acd260(59)=abb260(63)
      acd260(60)=dotproduct(ninjaE3,spvae1l5)
      acd260(61)=abb260(136)
      acd260(62)=dotproduct(ninjaE3,spvak2k1)
      acd260(63)=abb260(66)
      acd260(64)=dotproduct(ninjaA,spval5k2)
      acd260(65)=dotproduct(ninjaA,spval3k2)
      acd260(66)=dotproduct(ninjaA,spval5l3)
      acd260(67)=abb260(8)
      acd260(68)=abb260(68)
      acd260(69)=abb260(78)
      acd260(70)=abb260(49)
      acd260(71)=abb260(48)
      acd260(72)=abb260(77)
      acd260(73)=abb260(67)
      acd260(74)=abb260(39)
      acd260(75)=abb260(13)
      acd260(76)=abb260(54)
      acd260(77)=dotproduct(ninjaE3,spvak1e1)
      acd260(78)=abb260(35)
      acd260(79)=abb260(123)
      acd260(80)=dotproduct(ninjaE3,spval4e1)
      acd260(81)=abb260(112)
      acd260(82)=abb260(18)
      acd260(83)=abb260(23)
      acd260(84)=abb260(30)
      acd260(85)=abb260(60)
      acd260(86)=abb260(55)
      acd260(87)=abb260(57)
      acd260(88)=abb260(16)
      acd260(89)=abb260(129)
      acd260(90)=abb260(109)
      acd260(91)=dotproduct(ninjaE3,spvae1k1)
      acd260(92)=abb260(22)
      acd260(93)=abb260(122)
      acd260(94)=abb260(62)
      acd260(95)=abb260(99)
      acd260(96)=abb260(47)
      acd260(97)=abb260(141)
      acd260(98)=abb260(134)
      acd260(99)=abb260(31)
      acd260(100)=abb260(41)
      acd260(101)=acd260(8)*acd260(1)
      acd260(102)=acd260(101)*acd260(9)
      acd260(103)=acd260(6)*acd260(1)
      acd260(104)=acd260(103)*acd260(7)
      acd260(105)=acd260(2)*acd260(1)
      acd260(106)=acd260(105)*acd260(3)
      acd260(107)=acd260(4)*acd260(1)
      acd260(108)=acd260(107)*acd260(5)
      acd260(109)=acd260(13)*acd260(12)
      acd260(110)=acd260(109)*acd260(4)
      acd260(111)=acd260(11)*acd260(10)
      acd260(112)=acd260(111)*acd260(6)
      acd260(102)=acd260(102)+acd260(104)+acd260(106)+acd260(108)+acd260(110)+a&
      &cd260(112)
      acd260(104)=acd260(16)*acd260(102)
      acd260(106)=acd260(101)*acd260(19)
      acd260(108)=acd260(107)*acd260(18)
      acd260(110)=acd260(111)*acd260(25)
      acd260(106)=-acd260(110)+acd260(106)+acd260(108)
      acd260(108)=acd260(17)*acd260(106)
      acd260(110)=acd260(103)*acd260(23)
      acd260(112)=acd260(105)*acd260(22)
      acd260(113)=acd260(109)*acd260(28)
      acd260(110)=-acd260(113)+acd260(110)+acd260(112)
      acd260(112)=acd260(21)*acd260(110)
      acd260(113)=acd260(15)*acd260(14)
      acd260(114)=acd260(12)*acd260(4)
      acd260(115)=acd260(114)*acd260(113)
      acd260(116)=acd260(20)*acd260(4)*acd260(105)
      acd260(117)=acd260(101)*acd260(24)
      acd260(118)=acd260(6)*acd260(117)
      acd260(119)=acd260(27)*acd260(26)
      acd260(120)=acd260(119)*acd260(10)
      acd260(121)=acd260(2)*acd260(120)
      acd260(104)=acd260(121)+acd260(118)+acd260(116)+acd260(115)+2.0_ki*acd260&
      &(104)+acd260(112)+acd260(108)
      acd260(108)=acd260(39)*acd260(1)
      acd260(112)=acd260(50)*acd260(17)
      acd260(115)=acd260(51)*acd260(10)
      acd260(116)=acd260(52)*acd260(2)
      acd260(118)=acd260(53)*acd260(4)
      acd260(121)=acd260(54)*acd260(6)
      acd260(122)=acd260(55)*acd260(12)
      acd260(123)=acd260(56)*acd260(21)
      acd260(124)=acd260(58)*acd260(57)
      acd260(125)=acd260(59)*acd260(8)
      acd260(126)=acd260(61)*acd260(60)
      acd260(127)=acd260(63)*acd260(62)
      acd260(108)=acd260(127)+acd260(126)+acd260(125)+acd260(124)+acd260(123)+a&
      &cd260(122)+acd260(121)+acd260(118)+acd260(116)+acd260(115)+acd260(112)+a&
      &cd260(108)
      acd260(112)=2.0_ki*acd260(16)
      acd260(108)=acd260(112)*acd260(108)
      acd260(115)=acd260(34)*acd260(4)
      acd260(116)=acd260(30)*acd260(1)
      acd260(116)=acd260(115)+acd260(116)
      acd260(116)=acd260(18)*acd260(116)
      acd260(118)=acd260(19)*acd260(34)
      acd260(118)=acd260(70)+acd260(118)
      acd260(118)=acd260(8)*acd260(118)
      acd260(121)=acd260(42)*acd260(1)
      acd260(122)=-acd260(66)*acd260(111)
      acd260(123)=acd260(67)*acd260(10)
      acd260(124)=acd260(68)*acd260(4)
      acd260(125)=acd260(71)*acd260(25)
      acd260(126)=acd260(72)*acd260(60)
      acd260(127)=acd260(73)*acd260(62)
      acd260(116)=acd260(116)+acd260(127)+acd260(126)+acd260(125)+acd260(124)+a&
      &cd260(123)+acd260(122)+acd260(121)+acd260(118)
      acd260(116)=acd260(17)*acd260(116)
      acd260(118)=acd260(22)*acd260(2)
      acd260(121)=acd260(23)*acd260(6)
      acd260(118)=acd260(121)+acd260(118)
      acd260(118)=acd260(34)*acd260(118)
      acd260(121)=acd260(46)*acd260(1)
      acd260(122)=-acd260(65)*acd260(109)
      acd260(123)=acd260(69)*acd260(17)
      acd260(124)=acd260(83)*acd260(2)
      acd260(125)=acd260(89)*acd260(6)
      acd260(126)=acd260(93)*acd260(12)
      acd260(127)=acd260(99)*acd260(57)
      acd260(128)=acd260(100)*acd260(28)
      acd260(118)=acd260(128)+acd260(127)+acd260(126)+acd260(125)+acd260(124)+a&
      &cd260(123)+acd260(122)+acd260(121)+acd260(118)
      acd260(118)=acd260(21)*acd260(118)
      acd260(121)=acd260(30)*acd260(113)
      acd260(122)=acd260(92)*acd260(91)
      acd260(123)=acd260(94)*acd260(26)
      acd260(124)=acd260(95)*acd260(8)
      acd260(125)=acd260(96)*acd260(25)
      acd260(126)=acd260(97)*acd260(60)
      acd260(127)=acd260(98)*acd260(62)
      acd260(121)=acd260(127)+acd260(126)+acd260(125)+acd260(124)+acd260(123)+a&
      &cd260(122)+acd260(121)
      acd260(121)=acd260(12)*acd260(121)
      acd260(122)=acd260(64)*acd260(27)
      acd260(122)=acd260(74)+acd260(122)
      acd260(122)=acd260(2)*acd260(122)
      acd260(123)=acd260(32)*acd260(14)
      acd260(124)=acd260(75)*acd260(6)
      acd260(125)=-acd260(76)*acd260(57)
      acd260(126)=acd260(78)*acd260(77)
      acd260(127)=acd260(79)*acd260(28)
      acd260(128)=acd260(81)*acd260(80)
      acd260(122)=acd260(128)+acd260(127)+acd260(126)+acd260(125)+acd260(124)+a&
      &cd260(123)+acd260(122)
      acd260(122)=acd260(10)*acd260(122)
      acd260(123)=ninjaP+acd260(35)
      acd260(123)=acd260(102)*acd260(123)
      acd260(124)=acd260(82)*acd260(4)
      acd260(125)=acd260(84)*acd260(26)
      acd260(126)=acd260(85)*acd260(8)
      acd260(127)=acd260(86)*acd260(60)
      acd260(128)=acd260(87)*acd260(62)
      acd260(124)=acd260(128)+acd260(127)+acd260(126)+acd260(125)+acd260(124)
      acd260(124)=acd260(2)*acd260(124)
      acd260(125)=acd260(9)*acd260(8)
      acd260(126)=acd260(3)*acd260(2)
      acd260(127)=acd260(5)*acd260(4)
      acd260(128)=acd260(7)*acd260(6)
      acd260(125)=acd260(125)+acd260(128)+acd260(126)+acd260(127)
      acd260(125)=acd260(125)*acd260(34)*acd260(16)
      acd260(109)=acd260(16)*acd260(109)
      acd260(126)=acd260(16)*acd260(1)
      acd260(127)=acd260(5)*acd260(126)
      acd260(109)=acd260(109)+acd260(127)
      acd260(109)=acd260(30)*acd260(109)
      acd260(109)=acd260(125)+acd260(109)
      acd260(125)=acd260(3)*acd260(112)
      acd260(127)=acd260(22)*acd260(21)
      acd260(125)=acd260(127)+acd260(125)
      acd260(125)=acd260(1)*acd260(125)
      acd260(127)=acd260(20)*acd260(107)
      acd260(120)=acd260(120)+acd260(127)+acd260(125)
      acd260(120)=acd260(36)*acd260(120)
      acd260(111)=acd260(16)*acd260(111)
      acd260(125)=acd260(7)*acd260(126)
      acd260(111)=acd260(111)+acd260(125)
      acd260(125)=acd260(23)*acd260(21)*acd260(1)
      acd260(111)=acd260(117)+2.0_ki*acd260(111)+acd260(125)
      acd260(111)=acd260(37)*acd260(111)
      acd260(106)=acd260(40)*acd260(106)
      acd260(110)=acd260(41)*acd260(110)
      acd260(117)=acd260(29)*acd260(15)
      acd260(117)=acd260(88)+acd260(117)
      acd260(114)=acd260(114)*acd260(117)
      acd260(115)=acd260(2)*acd260(115)
      acd260(117)=acd260(30)*acd260(105)
      acd260(115)=acd260(115)+acd260(117)
      acd260(115)=acd260(20)*acd260(115)
      acd260(117)=acd260(4)*acd260(112)
      acd260(125)=-acd260(28)*acd260(21)
      acd260(117)=acd260(117)+acd260(125)
      acd260(117)=acd260(13)*acd260(117)
      acd260(113)=acd260(4)*acd260(113)
      acd260(113)=acd260(113)+acd260(117)
      acd260(113)=acd260(31)*acd260(113)
      acd260(117)=acd260(9)*acd260(112)
      acd260(125)=acd260(19)*acd260(17)
      acd260(117)=acd260(117)+acd260(125)
      acd260(117)=acd260(1)*acd260(117)
      acd260(125)=acd260(24)*acd260(103)
      acd260(117)=acd260(125)+acd260(117)
      acd260(117)=acd260(38)*acd260(117)
      acd260(112)=acd260(6)*acd260(112)
      acd260(125)=-acd260(25)*acd260(17)
      acd260(112)=acd260(112)+acd260(125)
      acd260(112)=acd260(11)*acd260(112)
      acd260(119)=acd260(119)*acd260(2)
      acd260(112)=acd260(119)+acd260(112)
      acd260(112)=acd260(49)*acd260(112)
      acd260(119)=acd260(24)*acd260(34)
      acd260(119)=-acd260(90)+acd260(119)
      acd260(119)=acd260(8)*acd260(6)*acd260(119)
      acd260(125)=-acd260(33)*acd260(14)*acd260(4)
      acd260(105)=acd260(43)*acd260(105)
      acd260(107)=acd260(44)*acd260(107)
      acd260(103)=acd260(45)*acd260(103)
      acd260(101)=acd260(47)*acd260(101)
      acd260(126)=acd260(48)*acd260(16)**2
      acd260(101)=4.0_ki*acd260(126)+acd260(101)+acd260(103)+acd260(107)+acd260&
      &(105)+acd260(125)+acd260(112)+acd260(110)+acd260(106)+acd260(117)+acd260&
      &(113)+acd260(111)+acd260(120)+acd260(115)+acd260(108)+acd260(118)+acd260&
      &(122)+acd260(121)+acd260(124)+acd260(114)+acd260(119)+acd260(123)+2.0_ki&
      &*acd260(109)+acd260(116)
      brack(ninjaidxt1mu0)=acd260(104)
      brack(ninjaidxt0mu0)=acd260(101)
      brack(ninjaidxt0mu2)=acd260(102)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d260h4_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd260h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d260h4l131
