module     p2_gg_httbar_d73h12l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d73h12l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd73h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd73
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      brack(ninjaidxt1x0mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd73h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(63) :: acd73
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd73(1)=dotproduct(k2,ninjaE3)
      acd73(2)=dotproduct(ninjaE3,spvak2e2)
      acd73(3)=abb73(20)
      acd73(4)=dotproduct(ninjaE3,spvae2k2)
      acd73(5)=abb73(29)
      acd73(6)=dotproduct(ninjaA0,ninjaE3)
      acd73(7)=abb73(11)
      acd73(8)=dotproduct(ninjaE3,spvae2e1)
      acd73(9)=abb73(24)
      acd73(10)=dotproduct(ninjaE3,spvae2l4)
      acd73(11)=abb73(25)
      acd73(12)=dotproduct(ninjaE3,spval5e2)
      acd73(13)=abb73(35)
      acd73(14)=abb73(13)
      acd73(15)=dotproduct(ninjaE3,spvae2k1)
      acd73(16)=abb73(22)
      acd73(17)=dotproduct(ninjaE3,spvae1e2)
      acd73(18)=abb73(21)
      acd73(19)=dotproduct(ninjaE3,spvae2l5)
      acd73(20)=abb73(42)
      acd73(21)=dotproduct(ninjaE3,spvak1e2)
      acd73(22)=abb73(39)
      acd73(23)=abb73(9)
      acd73(24)=abb73(34)
      acd73(25)=abb73(14)
      acd73(26)=dotproduct(ninjaE3,spvak1k2)
      acd73(27)=abb73(36)
      acd73(28)=abb73(37)
      acd73(29)=dotproduct(ninjaE3,spvae1k2)
      acd73(30)=abb73(38)
      acd73(31)=dotproduct(ninjaE3,spval5k2)
      acd73(32)=abb73(43)
      acd73(33)=dotproduct(ninjaE3,spvak2l3)
      acd73(34)=dotproduct(ninjaE3,spval3e2)
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
      acd73(55)=-acd73(7)*acd73(2)
      acd73(56)=acd73(9)*acd73(8)
      acd73(57)=acd73(11)*acd73(10)
      acd73(58)=acd73(13)*acd73(12)
      acd73(59)=-acd73(14)*acd73(4)
      acd73(60)=acd73(16)*acd73(15)
      acd73(61)=acd73(18)*acd73(17)
      acd73(62)=-acd73(20)*acd73(19)
      acd73(63)=-acd73(22)*acd73(21)
      acd73(55)=acd73(63)+acd73(62)+acd73(61)+acd73(60)+acd73(59)+acd73(58)+acd&
      &73(57)+acd73(55)+acd73(56)
      acd73(55)=acd73(6)*acd73(55)
      acd73(56)=acd73(3)*acd73(1)
      acd73(57)=acd73(23)*acd73(8)
      acd73(58)=acd73(24)*acd73(4)
      acd73(59)=acd73(25)*acd73(15)
      acd73(60)=acd73(27)*acd73(26)
      acd73(61)=acd73(28)*acd73(19)
      acd73(62)=acd73(30)*acd73(29)
      acd73(63)=acd73(32)*acd73(31)
      acd73(56)=acd73(63)+acd73(62)+acd73(61)+acd73(60)+acd73(59)+acd73(58)+acd&
      &73(57)+acd73(56)
      acd73(56)=acd73(2)*acd73(56)
      acd73(57)=acd73(5)*acd73(1)
      acd73(58)=acd73(42)*acd73(41)
      acd73(59)=acd73(44)*acd73(43)
      acd73(60)=acd73(46)*acd73(45)
      acd73(61)=acd73(48)*acd73(47)
      acd73(57)=acd73(61)+acd73(60)+acd73(59)+acd73(58)+acd73(57)
      acd73(57)=acd73(4)*acd73(57)
      acd73(58)=-acd73(49)*acd73(14)
      acd73(59)=acd73(51)*acd73(16)
      acd73(60)=acd73(52)*acd73(9)
      acd73(61)=acd73(53)*acd73(11)
      acd73(62)=-acd73(54)*acd73(20)
      acd73(58)=acd73(62)+acd73(61)+acd73(60)+acd73(59)+acd73(58)
      acd73(58)=acd73(50)*acd73(58)
      acd73(59)=-acd73(33)*acd73(7)
      acd73(60)=acd73(35)*acd73(18)
      acd73(61)=acd73(36)*acd73(13)
      acd73(62)=-acd73(37)*acd73(22)
      acd73(59)=acd73(62)+acd73(61)+acd73(60)+acd73(59)
      acd73(59)=acd73(34)*acd73(59)
      acd73(60)=acd73(38)*acd73(12)
      acd73(61)=acd73(39)*acd73(17)
      acd73(62)=acd73(40)*acd73(21)
      acd73(60)=acd73(62)+acd73(61)+acd73(60)
      acd73(60)=acd73(10)*acd73(60)
      acd73(55)=2.0_ki*acd73(55)+acd73(56)+acd73(58)+acd73(57)+acd73(59)+acd73(&
      &60)
      brack(ninjaidxt0x0mu0)=acd73(55)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d73h12_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd73h12
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k4
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
end module     p2_gg_httbar_d73h12l132
