module     p2_gg_httbar_d133h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d133h0l131.f90
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
      use p2_gg_httbar_abbrevd133h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(28) :: acd133
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd133(1)=dotproduct(k2,ninjaE3)
      acd133(2)=dotproduct(e2,ninjaE3)
      acd133(3)=abb133(65)
      acd133(4)=dotproduct(l5,ninjaE3)
      acd133(5)=abb133(57)
      acd133(6)=dotproduct(ninjaE3,spval5k1)
      acd133(7)=abb133(12)
      acd133(8)=dotproduct(ninjaE3,spval5k2)
      acd133(9)=abb133(15)
      acd133(10)=dotproduct(ninjaE3,spvak1k2)
      acd133(11)=abb133(22)
      acd133(12)=dotproduct(ninjaE3,spvae1k2)
      acd133(13)=abb133(24)
      acd133(14)=dotproduct(ninjaE3,spval4k2)
      acd133(15)=abb133(25)
      acd133(16)=dotproduct(ninjaE3,spval5l4)
      acd133(17)=abb133(30)
      acd133(18)=dotproduct(ninjaE3,spval5e1)
      acd133(19)=abb133(86)
      acd133(20)=acd133(3)*acd133(1)
      acd133(21)=acd133(5)*acd133(4)
      acd133(22)=acd133(7)*acd133(6)
      acd133(23)=acd133(9)*acd133(8)
      acd133(24)=acd133(11)*acd133(10)
      acd133(25)=acd133(13)*acd133(12)
      acd133(26)=acd133(15)*acd133(14)
      acd133(27)=acd133(17)*acd133(16)
      acd133(28)=acd133(19)*acd133(18)
      acd133(20)=acd133(28)+acd133(27)+acd133(26)+acd133(25)+acd133(24)+acd133(&
      &23)+acd133(22)+acd133(20)+acd133(21)
      acd133(20)=acd133(2)*acd133(20)
      brack(ninjaidxt2mu0)=acd133(20)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd133h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(129) :: acd133
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd133(1)=dotproduct(k2,ninjaE3)
      acd133(2)=dotproduct(e2,ninjaE4)
      acd133(3)=abb133(65)
      acd133(4)=dotproduct(k2,ninjaE4)
      acd133(5)=dotproduct(e2,ninjaE3)
      acd133(6)=dotproduct(l5,ninjaE3)
      acd133(7)=abb133(57)
      acd133(8)=dotproduct(l5,ninjaE4)
      acd133(9)=dotproduct(ninjaE4,spval5k1)
      acd133(10)=abb133(12)
      acd133(11)=dotproduct(ninjaE4,spval5k2)
      acd133(12)=abb133(15)
      acd133(13)=dotproduct(ninjaE4,spvak1k2)
      acd133(14)=abb133(22)
      acd133(15)=dotproduct(ninjaE4,spvae1k2)
      acd133(16)=abb133(24)
      acd133(17)=dotproduct(ninjaE4,spval4k2)
      acd133(18)=abb133(25)
      acd133(19)=dotproduct(ninjaE4,spval5l4)
      acd133(20)=abb133(30)
      acd133(21)=dotproduct(ninjaE4,spval5e1)
      acd133(22)=abb133(86)
      acd133(23)=dotproduct(ninjaE3,spval5k1)
      acd133(24)=dotproduct(ninjaE3,spval5k2)
      acd133(25)=dotproduct(ninjaE3,spvak1k2)
      acd133(26)=dotproduct(ninjaE3,spvae1k2)
      acd133(27)=dotproduct(ninjaE3,spval4k2)
      acd133(28)=dotproduct(ninjaE3,spval5l4)
      acd133(29)=dotproduct(ninjaE3,spval5e1)
      acd133(30)=abb133(105)
      acd133(31)=dotproduct(k2,ninjaA)
      acd133(32)=dotproduct(e2,ninjaA)
      acd133(33)=abb133(13)
      acd133(34)=dotproduct(l5,ninjaA)
      acd133(35)=abb133(38)
      acd133(36)=dotproduct(ninjaA,spval5k1)
      acd133(37)=dotproduct(ninjaA,spval5k2)
      acd133(38)=dotproduct(ninjaA,spvak1k2)
      acd133(39)=dotproduct(ninjaA,spvae1k2)
      acd133(40)=dotproduct(ninjaA,spval4k2)
      acd133(41)=dotproduct(ninjaA,spval5l4)
      acd133(42)=dotproduct(ninjaA,spval5e1)
      acd133(43)=abb133(43)
      acd133(44)=dotproduct(ninjaA,ninjaE3)
      acd133(45)=dotproduct(ninjaE3,spvak2l5)
      acd133(46)=abb133(11)
      acd133(47)=abb133(17)
      acd133(48)=dotproduct(ninjaE3,spvae2k2)
      acd133(49)=abb133(14)
      acd133(50)=abb133(18)
      acd133(51)=dotproduct(ninjaE3,spvak2e2)
      acd133(52)=abb133(16)
      acd133(53)=dotproduct(ninjaE3,spvak2l4)
      acd133(54)=abb133(19)
      acd133(55)=dotproduct(ninjaE3,spvak1l5)
      acd133(56)=abb133(20)
      acd133(57)=dotproduct(ninjaE3,spval4l5)
      acd133(58)=abb133(21)
      acd133(59)=dotproduct(ninjaE3,spvak2k1)
      acd133(60)=abb133(23)
      acd133(61)=dotproduct(ninjaE3,spvae1l5)
      acd133(62)=abb133(26)
      acd133(63)=abb133(55)
      acd133(64)=dotproduct(ninjaE3,spvae2k1)
      acd133(65)=abb133(35)
      acd133(66)=dotproduct(ninjaE3,spvak2e1)
      acd133(67)=abb133(39)
      acd133(68)=abb133(41)
      acd133(69)=dotproduct(ninjaE3,spval5e2)
      acd133(70)=abb133(46)
      acd133(71)=dotproduct(ninjaE3,spvae1e2)
      acd133(72)=abb133(47)
      acd133(73)=dotproduct(ninjaE3,spvak1e2)
      acd133(74)=abb133(50)
      acd133(75)=dotproduct(ninjaE3,spvae2l5)
      acd133(76)=abb133(69)
      acd133(77)=dotproduct(ninjaE3,spvae2e1)
      acd133(78)=abb133(94)
      acd133(79)=dotproduct(ninjaE3,spvae2l4)
      acd133(80)=abb133(120)
      acd133(81)=dotproduct(ninjaE3,spval4e2)
      acd133(82)=abb133(125)
      acd133(83)=dotproduct(ninjaA,ninjaA)
      acd133(84)=dotproduct(ninjaA,spvak2l5)
      acd133(85)=dotproduct(ninjaA,spvae2k2)
      acd133(86)=dotproduct(ninjaA,spvak2e2)
      acd133(87)=dotproduct(ninjaA,spvak2l4)
      acd133(88)=dotproduct(ninjaA,spvak1l5)
      acd133(89)=dotproduct(ninjaA,spval4l5)
      acd133(90)=dotproduct(ninjaA,spvak2k1)
      acd133(91)=dotproduct(ninjaA,spvae1l5)
      acd133(92)=dotproduct(ninjaA,spvae2k1)
      acd133(93)=dotproduct(ninjaA,spvak2e1)
      acd133(94)=dotproduct(ninjaA,spval5e2)
      acd133(95)=dotproduct(ninjaA,spvae1e2)
      acd133(96)=dotproduct(ninjaA,spvak1e2)
      acd133(97)=dotproduct(ninjaA,spvae2l5)
      acd133(98)=dotproduct(ninjaA,spvae2e1)
      acd133(99)=dotproduct(ninjaA,spvae2l4)
      acd133(100)=dotproduct(ninjaA,spval4e2)
      acd133(101)=abb133(40)
      acd133(102)=acd133(22)*acd133(21)
      acd133(103)=acd133(20)*acd133(19)
      acd133(104)=acd133(18)*acd133(17)
      acd133(105)=acd133(16)*acd133(15)
      acd133(106)=acd133(14)*acd133(13)
      acd133(107)=acd133(12)*acd133(11)
      acd133(108)=acd133(10)*acd133(9)
      acd133(109)=acd133(7)*acd133(8)
      acd133(110)=acd133(3)*acd133(4)
      acd133(102)=acd133(106)+acd133(107)+acd133(108)+acd133(109)+acd133(102)+a&
      &cd133(104)+acd133(105)+acd133(103)+acd133(110)
      acd133(102)=acd133(102)*acd133(5)
      acd133(103)=acd133(22)*acd133(29)
      acd133(104)=acd133(20)*acd133(28)
      acd133(105)=acd133(18)*acd133(27)
      acd133(106)=acd133(16)*acd133(26)
      acd133(107)=acd133(14)*acd133(25)
      acd133(108)=acd133(12)*acd133(24)
      acd133(109)=acd133(10)*acd133(23)
      acd133(110)=acd133(7)*acd133(6)
      acd133(111)=acd133(3)*acd133(1)
      acd133(103)=acd133(107)+acd133(106)+acd133(105)+acd133(103)+acd133(104)+a&
      &cd133(108)+acd133(109)+acd133(110)+acd133(111)
      acd133(104)=acd133(103)*acd133(2)
      acd133(102)=acd133(102)+acd133(104)-acd133(30)
      acd133(103)=acd133(32)*acd133(103)
      acd133(104)=acd133(22)*acd133(42)
      acd133(105)=acd133(20)*acd133(41)
      acd133(106)=acd133(18)*acd133(40)
      acd133(107)=acd133(16)*acd133(39)
      acd133(108)=acd133(14)*acd133(38)
      acd133(109)=acd133(12)*acd133(37)
      acd133(110)=acd133(10)*acd133(36)
      acd133(111)=acd133(7)*acd133(34)
      acd133(112)=acd133(3)*acd133(31)
      acd133(104)=acd133(110)+acd133(111)+acd133(112)+acd133(43)+acd133(106)+ac&
      &d133(107)+acd133(108)+acd133(109)+acd133(104)+acd133(105)
      acd133(105)=acd133(5)*acd133(104)
      acd133(106)=acd133(82)*acd133(81)
      acd133(107)=acd133(80)*acd133(79)
      acd133(108)=acd133(78)*acd133(77)
      acd133(109)=acd133(76)*acd133(75)
      acd133(110)=acd133(74)*acd133(73)
      acd133(111)=acd133(72)*acd133(71)
      acd133(112)=acd133(70)*acd133(69)
      acd133(113)=acd133(67)*acd133(66)
      acd133(114)=acd133(65)*acd133(64)
      acd133(115)=acd133(62)*acd133(61)
      acd133(116)=acd133(60)*acd133(59)
      acd133(117)=acd133(58)*acd133(57)
      acd133(118)=acd133(56)*acd133(55)
      acd133(119)=acd133(54)*acd133(53)
      acd133(120)=acd133(52)*acd133(51)
      acd133(121)=acd133(49)*acd133(48)
      acd133(122)=acd133(46)*acd133(45)
      acd133(123)=acd133(30)*acd133(44)
      acd133(124)=acd133(29)*acd133(68)
      acd133(125)=acd133(28)*acd133(63)
      acd133(126)=acd133(24)*acd133(50)
      acd133(127)=acd133(23)*acd133(47)
      acd133(128)=acd133(6)*acd133(35)
      acd133(129)=acd133(1)*acd133(33)
      acd133(103)=acd133(105)+acd133(103)+acd133(129)+acd133(128)+acd133(127)+a&
      &cd133(126)+acd133(125)+acd133(124)-2.0_ki*acd133(123)+acd133(122)+acd133&
      &(121)+acd133(120)+acd133(119)+acd133(118)+acd133(117)+acd133(116)+acd133&
      &(115)+acd133(114)+acd133(113)+acd133(112)+acd133(111)+acd133(110)+acd133&
      &(109)+acd133(108)+acd133(106)+acd133(107)
      acd133(105)=ninjaP*acd133(102)
      acd133(104)=acd133(32)*acd133(104)
      acd133(106)=acd133(82)*acd133(100)
      acd133(107)=acd133(80)*acd133(99)
      acd133(108)=acd133(78)*acd133(98)
      acd133(109)=acd133(76)*acd133(97)
      acd133(110)=acd133(74)*acd133(96)
      acd133(111)=acd133(72)*acd133(95)
      acd133(112)=acd133(70)*acd133(94)
      acd133(113)=acd133(67)*acd133(93)
      acd133(114)=acd133(65)*acd133(92)
      acd133(115)=acd133(62)*acd133(91)
      acd133(116)=acd133(60)*acd133(90)
      acd133(117)=acd133(58)*acd133(89)
      acd133(118)=acd133(56)*acd133(88)
      acd133(119)=acd133(54)*acd133(87)
      acd133(120)=acd133(52)*acd133(86)
      acd133(121)=acd133(49)*acd133(85)
      acd133(122)=acd133(46)*acd133(84)
      acd133(123)=acd133(42)*acd133(68)
      acd133(124)=acd133(41)*acd133(63)
      acd133(125)=acd133(37)*acd133(50)
      acd133(126)=acd133(36)*acd133(47)
      acd133(127)=acd133(34)*acd133(35)
      acd133(128)=acd133(31)*acd133(33)
      acd133(129)=-acd133(30)*acd133(83)
      acd133(104)=acd133(104)+acd133(129)+acd133(128)+acd133(127)+acd133(126)+a&
      &cd133(125)+acd133(124)+acd133(123)+acd133(122)+acd133(121)+acd133(120)+a&
      &cd133(119)+acd133(118)+acd133(117)+acd133(116)+acd133(115)+acd133(114)+a&
      &cd133(113)+acd133(112)+acd133(111)+acd133(110)+acd133(109)+acd133(108)+a&
      &cd133(107)+acd133(101)+acd133(106)+acd133(105)
      brack(ninjaidxt1mu0)=acd133(103)
      brack(ninjaidxt0mu0)=acd133(104)
      brack(ninjaidxt0mu2)=acd133(102)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d133h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd133h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA(1:4) = + a(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d133h0l131
