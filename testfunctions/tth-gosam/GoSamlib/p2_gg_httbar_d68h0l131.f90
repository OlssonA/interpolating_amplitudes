module     p2_gg_httbar_d68h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d68h0l131.f90
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
      use p2_gg_httbar_abbrevd68h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd68
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd68h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(63) :: acd68
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd68(1)=dotproduct(ninjaE3,spval4k2)
      acd68(2)=abb68(28)
      acd68(3)=dotproduct(ninjaE3,spval4k1)
      acd68(4)=abb68(26)
      acd68(5)=dotproduct(ninjaE3,spval5k2)
      acd68(6)=abb68(45)
      acd68(7)=dotproduct(ninjaE3,spval5k1)
      acd68(8)=abb68(38)
      acd68(9)=dotproduct(k2,ninjaE3)
      acd68(10)=abb68(25)
      acd68(11)=abb68(48)
      acd68(12)=dotproduct(ninjaA,ninjaE3)
      acd68(13)=dotproduct(ninjaE3,spvak1k2)
      acd68(14)=abb68(22)
      acd68(15)=abb68(47)
      acd68(16)=dotproduct(k2,ninjaA)
      acd68(17)=dotproduct(ninjaA,spval4k2)
      acd68(18)=dotproduct(ninjaA,spval5k2)
      acd68(19)=abb68(13)
      acd68(20)=dotproduct(l4,ninjaE3)
      acd68(21)=abb68(16)
      acd68(22)=dotproduct(l5,ninjaE3)
      acd68(23)=abb68(20)
      acd68(24)=dotproduct(ninjaA,ninjaA)
      acd68(25)=dotproduct(ninjaA,spval4k1)
      acd68(26)=dotproduct(ninjaA,spval5k1)
      acd68(27)=abb68(10)
      acd68(28)=dotproduct(ninjaA,spvak1k2)
      acd68(29)=abb68(11)
      acd68(30)=dotproduct(ninjaE3,spval4l5)
      acd68(31)=abb68(14)
      acd68(32)=dotproduct(ninjaE3,spval3k2)
      acd68(33)=abb68(15)
      acd68(34)=abb68(17)
      acd68(35)=abb68(18)
      acd68(36)=dotproduct(ninjaE3,spval4l3)
      acd68(37)=abb68(24)
      acd68(38)=abb68(29)
      acd68(39)=abb68(30)
      acd68(40)=dotproduct(ninjaE3,spval5l4)
      acd68(41)=abb68(32)
      acd68(42)=acd68(2)*acd68(1)
      acd68(43)=acd68(4)*acd68(3)
      acd68(44)=acd68(6)*acd68(5)
      acd68(45)=acd68(8)*acd68(7)
      acd68(42)=acd68(42)+acd68(43)+acd68(44)+acd68(45)
      acd68(43)=acd68(12)*acd68(42)
      acd68(44)=acd68(10)*acd68(9)
      acd68(45)=acd68(1)*acd68(44)
      acd68(46)=acd68(11)*acd68(9)
      acd68(47)=acd68(5)*acd68(46)
      acd68(48)=acd68(14)*acd68(13)
      acd68(49)=acd68(3)*acd68(48)
      acd68(50)=acd68(15)*acd68(13)
      acd68(51)=-acd68(7)*acd68(50)
      acd68(43)=acd68(51)+acd68(49)+acd68(47)+2.0_ki*acd68(43)+acd68(45)
      acd68(45)=acd68(24)+ninjaP
      acd68(45)=acd68(42)*acd68(45)
      acd68(47)=acd68(10)*acd68(1)
      acd68(49)=acd68(11)*acd68(5)
      acd68(47)=acd68(47)+acd68(49)
      acd68(47)=acd68(16)*acd68(47)
      acd68(49)=2.0_ki*acd68(12)
      acd68(51)=acd68(2)*acd68(49)
      acd68(44)=acd68(51)+acd68(44)
      acd68(44)=acd68(17)*acd68(44)
      acd68(51)=acd68(6)*acd68(49)
      acd68(46)=acd68(51)+acd68(46)
      acd68(46)=acd68(18)*acd68(46)
      acd68(51)=acd68(4)*acd68(49)
      acd68(48)=acd68(51)+acd68(48)
      acd68(48)=acd68(25)*acd68(48)
      acd68(51)=acd68(8)*acd68(49)
      acd68(50)=acd68(51)-acd68(50)
      acd68(50)=acd68(26)*acd68(50)
      acd68(51)=acd68(14)*acd68(3)
      acd68(52)=acd68(15)*acd68(7)
      acd68(51)=acd68(51)-acd68(52)
      acd68(51)=acd68(28)*acd68(51)
      acd68(52)=acd68(19)*acd68(9)
      acd68(53)=acd68(21)*acd68(20)
      acd68(54)=acd68(23)*acd68(22)
      acd68(49)=acd68(27)*acd68(49)
      acd68(55)=acd68(29)*acd68(13)
      acd68(56)=acd68(31)*acd68(30)
      acd68(57)=acd68(33)*acd68(32)
      acd68(58)=acd68(34)*acd68(1)
      acd68(59)=acd68(35)*acd68(3)
      acd68(60)=acd68(37)*acd68(36)
      acd68(61)=acd68(38)*acd68(5)
      acd68(62)=acd68(39)*acd68(7)
      acd68(63)=acd68(41)*acd68(40)
      acd68(44)=acd68(63)+acd68(62)+acd68(61)+acd68(60)+acd68(59)+acd68(58)+acd&
      &68(57)+acd68(56)+acd68(55)+acd68(49)+acd68(54)+acd68(53)+acd68(52)+acd68&
      &(51)+acd68(50)+acd68(48)+acd68(46)+acd68(44)+acd68(47)+acd68(45)
      brack(ninjaidxt1mu0)=acd68(43)
      brack(ninjaidxt0mu0)=acd68(44)
      brack(ninjaidxt0mu2)=acd68(42)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d68h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd68h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k5
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
end module     p2_gg_httbar_d68h0l131
