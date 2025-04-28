module     p2_gg_httbar_d73h12l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d73h12l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd73h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd73
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd73h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(124) :: acd73
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd73(1)=dotproduct(ninjaE3,spvak2e2)
      acd73(2)=abb73(11)
      acd73(3)=dotproduct(ninjaE3,spvae2e1)
      acd73(4)=abb73(24)
      acd73(5)=dotproduct(ninjaE3,spvae2l4)
      acd73(6)=abb73(25)
      acd73(7)=dotproduct(ninjaE3,spval5e2)
      acd73(8)=abb73(35)
      acd73(9)=dotproduct(ninjaE3,spvae2k2)
      acd73(10)=abb73(13)
      acd73(11)=dotproduct(ninjaE3,spvae2k1)
      acd73(12)=abb73(22)
      acd73(13)=dotproduct(ninjaE3,spvae1e2)
      acd73(14)=abb73(21)
      acd73(15)=dotproduct(ninjaE3,spvae2l5)
      acd73(16)=abb73(42)
      acd73(17)=dotproduct(ninjaE3,spvak1e2)
      acd73(18)=abb73(39)
      acd73(19)=dotproduct(k2,ninjaE3)
      acd73(20)=abb73(20)
      acd73(21)=abb73(29)
      acd73(22)=dotproduct(ninjaA,ninjaE3)
      acd73(23)=abb73(9)
      acd73(24)=abb73(34)
      acd73(25)=abb73(14)
      acd73(26)=abb73(37)
      acd73(27)=dotproduct(ninjaE3,spvak1k2)
      acd73(28)=abb73(36)
      acd73(29)=dotproduct(ninjaE3,spvae1k2)
      acd73(30)=abb73(38)
      acd73(31)=dotproduct(ninjaE3,spval5k2)
      acd73(32)=abb73(43)
      acd73(33)=dotproduct(ninjaE3,spval3e2)
      acd73(34)=dotproduct(ninjaE3,spvak2l3)
      acd73(35)=dotproduct(ninjaE3,spvae1l3)
      acd73(36)=dotproduct(ninjaE3,spval5l3)
      acd73(37)=dotproduct(ninjaE3,spvak1l3)
      acd73(38)=abb73(12)
      acd73(39)=abb73(26)
      acd73(40)=abb73(41)
      acd73(41)=dotproduct(ninjaE3,spvak2e1)
      acd73(42)=abb73(31)
      acd73(43)=dotproduct(ninjaE3,spvak2l5)
      acd73(44)=abb73(44)
      acd73(45)=dotproduct(ninjaE3,spvak2l4)
      acd73(46)=abb73(45)
      acd73(47)=dotproduct(ninjaE3,spvak2k1)
      acd73(48)=abb73(46)
      acd73(49)=dotproduct(ninjaE3,spval3k2)
      acd73(50)=dotproduct(ninjaE3,spvae2l3)
      acd73(51)=dotproduct(ninjaE3,spval3k1)
      acd73(52)=dotproduct(ninjaE3,spval3e1)
      acd73(53)=dotproduct(ninjaE3,spval3l4)
      acd73(54)=dotproduct(ninjaE3,spval3l5)
      acd73(55)=dotproduct(k2,ninjaA)
      acd73(56)=dotproduct(ninjaA,spvak2e2)
      acd73(57)=dotproduct(ninjaA,spvae2k2)
      acd73(58)=dotproduct(ninjaA,ninjaA)
      acd73(59)=dotproduct(ninjaA,spvae2e1)
      acd73(60)=dotproduct(ninjaA,spvae2l4)
      acd73(61)=dotproduct(ninjaA,spval5e2)
      acd73(62)=dotproduct(ninjaA,spvae2k1)
      acd73(63)=dotproduct(ninjaA,spvae1e2)
      acd73(64)=dotproduct(ninjaA,spvae2l5)
      acd73(65)=dotproduct(ninjaA,spvak1e2)
      acd73(66)=abb73(18)
      acd73(67)=dotproduct(ninjaA,spval3e2)
      acd73(68)=dotproduct(ninjaA,spvak2l3)
      acd73(69)=dotproduct(ninjaA,spval3k2)
      acd73(70)=dotproduct(ninjaA,spvae2l3)
      acd73(71)=dotproduct(ninjaA,spvae1l3)
      acd73(72)=dotproduct(ninjaA,spval3k1)
      acd73(73)=dotproduct(ninjaA,spval3e1)
      acd73(74)=dotproduct(ninjaA,spval3l4)
      acd73(75)=dotproduct(ninjaA,spvak2e1)
      acd73(76)=dotproduct(ninjaA,spval5l3)
      acd73(77)=dotproduct(ninjaA,spvak1k2)
      acd73(78)=dotproduct(ninjaA,spvae1k2)
      acd73(79)=dotproduct(ninjaA,spvak1l3)
      acd73(80)=dotproduct(ninjaA,spval3l5)
      acd73(81)=dotproduct(ninjaA,spval5k2)
      acd73(82)=dotproduct(ninjaA,spvak2l5)
      acd73(83)=dotproduct(ninjaA,spvak2l4)
      acd73(84)=dotproduct(ninjaA,spvak2k1)
      acd73(85)=abb73(15)
      acd73(86)=abb73(19)
      acd73(87)=abb73(10)
      acd73(88)=abb73(27)
      acd73(89)=abb73(33)
      acd73(90)=abb73(16)
      acd73(91)=abb73(23)
      acd73(92)=abb73(30)
      acd73(93)=abb73(28)
      acd73(94)=abb73(32)
      acd73(95)=abb73(40)
      acd73(96)=acd73(6)*acd73(5)
      acd73(97)=acd73(10)*acd73(9)
      acd73(98)=acd73(3)*acd73(4)
      acd73(99)=acd73(7)*acd73(8)
      acd73(100)=acd73(11)*acd73(12)
      acd73(101)=acd73(13)*acd73(14)
      acd73(102)=acd73(15)*acd73(16)
      acd73(103)=acd73(17)*acd73(18)
      acd73(96)=-acd73(103)+acd73(100)+acd73(98)+acd73(99)+acd73(101)-acd73(102&
      &)+acd73(96)-acd73(97)
      acd73(97)=acd73(2)*acd73(1)
      acd73(97)=acd73(97)-acd73(96)
      acd73(98)=-acd73(22)*acd73(97)
      acd73(99)=acd73(20)*acd73(19)
      acd73(100)=acd73(23)*acd73(3)
      acd73(101)=acd73(25)*acd73(11)
      acd73(102)=acd73(26)*acd73(15)
      acd73(103)=acd73(27)*acd73(28)
      acd73(104)=acd73(29)*acd73(30)
      acd73(105)=acd73(31)*acd73(32)
      acd73(99)=acd73(99)+acd73(103)+acd73(104)+acd73(100)+acd73(101)+acd73(102&
      &)+acd73(105)
      acd73(100)=acd73(24)*acd73(9)
      acd73(100)=acd73(100)+acd73(99)
      acd73(100)=acd73(1)*acd73(100)
      acd73(101)=acd73(21)*acd73(19)
      acd73(102)=acd73(41)*acd73(42)
      acd73(103)=acd73(43)*acd73(44)
      acd73(104)=acd73(45)*acd73(46)
      acd73(105)=acd73(47)*acd73(48)
      acd73(101)=acd73(101)+acd73(102)+acd73(103)+acd73(104)+acd73(105)
      acd73(102)=acd73(9)*acd73(101)
      acd73(103)=acd73(49)*acd73(10)
      acd73(104)=acd73(51)*acd73(12)
      acd73(105)=acd73(52)*acd73(4)
      acd73(106)=acd73(53)*acd73(6)
      acd73(107)=acd73(54)*acd73(16)
      acd73(103)=-acd73(107)-acd73(103)+acd73(104)+acd73(105)+acd73(106)
      acd73(104)=acd73(50)*acd73(103)
      acd73(105)=acd73(34)*acd73(2)
      acd73(106)=acd73(35)*acd73(14)
      acd73(107)=acd73(36)*acd73(8)
      acd73(108)=acd73(37)*acd73(18)
      acd73(105)=-acd73(105)+acd73(106)+acd73(107)-acd73(108)
      acd73(106)=acd73(33)*acd73(105)
      acd73(107)=acd73(38)*acd73(7)
      acd73(108)=acd73(39)*acd73(13)
      acd73(109)=acd73(40)*acd73(17)
      acd73(107)=acd73(107)+acd73(108)+acd73(109)
      acd73(108)=acd73(5)*acd73(107)
      acd73(98)=2.0_ki*acd73(98)+acd73(100)+acd73(104)+acd73(102)+acd73(106)+ac&
      &d73(108)
      acd73(100)=ninjaP+acd73(58)
      acd73(96)=acd73(100)*acd73(96)
      acd73(99)=acd73(56)*acd73(99)
      acd73(102)=acd73(70)*acd73(103)
      acd73(103)=-acd73(69)*acd73(10)
      acd73(104)=acd73(72)*acd73(12)
      acd73(106)=acd73(73)*acd73(4)
      acd73(108)=acd73(74)*acd73(6)
      acd73(109)=-acd73(80)*acd73(16)
      acd73(103)=acd73(91)+acd73(109)+acd73(108)+acd73(106)+acd73(104)+acd73(10&
      &3)
      acd73(103)=acd73(50)*acd73(103)
      acd73(104)=2.0_ki*acd73(22)
      acd73(106)=-acd73(10)*acd73(104)
      acd73(101)=acd73(106)+acd73(101)
      acd73(101)=acd73(57)*acd73(101)
      acd73(105)=acd73(67)*acd73(105)
      acd73(106)=acd73(75)*acd73(42)
      acd73(108)=acd73(82)*acd73(44)
      acd73(109)=acd73(83)*acd73(46)
      acd73(110)=acd73(84)*acd73(48)
      acd73(106)=acd73(90)+acd73(110)+acd73(109)+acd73(108)+acd73(106)
      acd73(106)=acd73(9)*acd73(106)
      acd73(108)=-acd73(68)*acd73(2)
      acd73(109)=acd73(71)*acd73(14)
      acd73(110)=acd73(76)*acd73(8)
      acd73(111)=-acd73(79)*acd73(18)
      acd73(108)=acd73(87)+acd73(111)+acd73(110)+acd73(109)+acd73(108)
      acd73(108)=acd73(33)*acd73(108)
      acd73(109)=acd73(77)*acd73(28)
      acd73(110)=acd73(78)*acd73(30)
      acd73(111)=acd73(81)*acd73(32)
      acd73(109)=acd73(85)+acd73(111)+acd73(110)+acd73(109)
      acd73(109)=acd73(1)*acd73(109)
      acd73(110)=acd73(6)*acd73(104)
      acd73(107)=acd73(110)+acd73(107)
      acd73(107)=acd73(60)*acd73(107)
      acd73(100)=-acd73(1)*acd73(100)
      acd73(110)=-acd73(56)*acd73(104)
      acd73(100)=acd73(110)+acd73(100)
      acd73(100)=acd73(2)*acd73(100)
      acd73(110)=acd73(56)*acd73(9)
      acd73(111)=acd73(57)*acd73(1)
      acd73(110)=acd73(110)+acd73(111)
      acd73(110)=acd73(24)*acd73(110)
      acd73(111)=acd73(20)*acd73(1)
      acd73(112)=acd73(21)*acd73(9)
      acd73(111)=acd73(111)+acd73(112)
      acd73(111)=acd73(55)*acd73(111)
      acd73(112)=acd73(4)*acd73(104)
      acd73(113)=acd73(23)*acd73(1)
      acd73(112)=acd73(112)+acd73(113)
      acd73(112)=acd73(59)*acd73(112)
      acd73(113)=acd73(8)*acd73(104)
      acd73(114)=acd73(38)*acd73(5)
      acd73(113)=acd73(113)+acd73(114)
      acd73(113)=acd73(61)*acd73(113)
      acd73(114)=acd73(12)*acd73(104)
      acd73(115)=acd73(25)*acd73(1)
      acd73(114)=acd73(114)+acd73(115)
      acd73(114)=acd73(62)*acd73(114)
      acd73(115)=acd73(14)*acd73(104)
      acd73(116)=acd73(39)*acd73(5)
      acd73(115)=acd73(115)+acd73(116)
      acd73(115)=acd73(63)*acd73(115)
      acd73(116)=-acd73(16)*acd73(104)
      acd73(117)=acd73(26)*acd73(1)
      acd73(116)=acd73(116)+acd73(117)
      acd73(116)=acd73(64)*acd73(116)
      acd73(117)=-acd73(18)*acd73(104)
      acd73(118)=acd73(40)*acd73(5)
      acd73(117)=acd73(117)+acd73(118)
      acd73(117)=acd73(65)*acd73(117)
      acd73(104)=acd73(66)*acd73(104)
      acd73(118)=acd73(86)*acd73(3)
      acd73(119)=acd73(88)*acd73(5)
      acd73(120)=acd73(89)*acd73(7)
      acd73(121)=acd73(92)*acd73(11)
      acd73(122)=acd73(93)*acd73(13)
      acd73(123)=acd73(94)*acd73(15)
      acd73(124)=acd73(95)*acd73(17)
      acd73(96)=acd73(124)+acd73(123)+acd73(122)+acd73(121)+acd73(120)+acd73(11&
      &9)+acd73(118)+acd73(104)+acd73(117)+acd73(116)+acd73(115)+acd73(114)+acd&
      &73(113)+acd73(112)+acd73(111)+acd73(110)+acd73(100)+acd73(99)+acd73(101)&
      &+acd73(103)+acd73(102)+acd73(108)+acd73(106)+acd73(105)+acd73(107)+acd73&
      &(109)+acd73(96)
      brack(ninjaidxt1mu0)=acd73(98)
      brack(ninjaidxt0mu0)=acd73(96)
      brack(ninjaidxt0mu2)=-acd73(97)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d73h12_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd73h12_qp
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
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d73h12l131_qp
