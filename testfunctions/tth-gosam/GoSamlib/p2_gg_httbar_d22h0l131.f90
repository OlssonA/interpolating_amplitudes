module     p2_gg_httbar_d22h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d22h0l131.f90
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
      use p2_gg_httbar_abbrevd22h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(13) :: acd22
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd22(1)=dotproduct(k2,ninjaE3)
      acd22(2)=dotproduct(ninjaE3,spval4k2)
      acd22(3)=abb22(10)
      acd22(4)=dotproduct(ninjaE3,spval5k2)
      acd22(5)=abb22(19)
      acd22(6)=dotproduct(ninjaE3,spvak1k2)
      acd22(7)=dotproduct(ninjaE3,spval4k1)
      acd22(8)=abb22(8)
      acd22(9)=dotproduct(ninjaE3,spval5k1)
      acd22(10)=abb22(9)
      acd22(11)=acd22(3)*acd22(2)
      acd22(12)=acd22(5)*acd22(4)
      acd22(11)=acd22(11)+acd22(12)
      acd22(11)=acd22(1)*acd22(11)
      acd22(12)=acd22(8)*acd22(7)
      acd22(13)=acd22(10)*acd22(9)
      acd22(12)=acd22(13)+acd22(12)
      acd22(12)=acd22(6)*acd22(12)
      acd22(11)=acd22(12)+acd22(11)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd22(11)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd22h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(55) :: acd22
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd22(1)=dotproduct(k2,ninjaE3)
      acd22(2)=dotproduct(ninjaE4,spval4k2)
      acd22(3)=abb22(10)
      acd22(4)=dotproduct(ninjaE4,spval5k2)
      acd22(5)=abb22(19)
      acd22(6)=dotproduct(k2,ninjaE4)
      acd22(7)=dotproduct(ninjaE3,spval4k2)
      acd22(8)=dotproduct(ninjaE3,spval5k2)
      acd22(9)=dotproduct(ninjaE3,spvak1k2)
      acd22(10)=dotproduct(ninjaE4,spval4k1)
      acd22(11)=abb22(8)
      acd22(12)=dotproduct(ninjaE4,spval5k1)
      acd22(13)=abb22(9)
      acd22(14)=dotproduct(ninjaE3,spval4k1)
      acd22(15)=dotproduct(ninjaE4,spvak1k2)
      acd22(16)=dotproduct(ninjaE3,spval5k1)
      acd22(17)=abb22(12)
      acd22(18)=dotproduct(k2,ninjaA)
      acd22(19)=dotproduct(ninjaA,spval4k2)
      acd22(20)=dotproduct(ninjaA,spval5k2)
      acd22(21)=abb22(18)
      acd22(22)=dotproduct(ninjaA,ninjaE3)
      acd22(23)=dotproduct(ninjaA,spvak1k2)
      acd22(24)=dotproduct(ninjaA,spval4k1)
      acd22(25)=dotproduct(ninjaA,spval5k1)
      acd22(26)=abb22(7)
      acd22(27)=abb22(14)
      acd22(28)=abb22(17)
      acd22(29)=abb22(20)
      acd22(30)=abb22(13)
      acd22(31)=dotproduct(ninjaE3,spval3k2)
      acd22(32)=abb22(15)
      acd22(33)=dotproduct(ninjaE3,spval3k1)
      acd22(34)=abb22(21)
      acd22(35)=dotproduct(ninjaE3,spvak1l3)
      acd22(36)=abb22(22)
      acd22(37)=dotproduct(ninjaA,ninjaA)
      acd22(38)=dotproduct(ninjaA,spval3k2)
      acd22(39)=dotproduct(ninjaA,spval3k1)
      acd22(40)=dotproduct(ninjaA,spvak1l3)
      acd22(41)=abb22(6)
      acd22(42)=acd22(13)*acd22(12)
      acd22(43)=acd22(11)*acd22(10)
      acd22(42)=acd22(42)+acd22(43)
      acd22(42)=acd22(42)*acd22(9)
      acd22(43)=acd22(5)*acd22(4)
      acd22(44)=acd22(3)*acd22(2)
      acd22(43)=acd22(43)+acd22(44)
      acd22(43)=acd22(43)*acd22(1)
      acd22(44)=acd22(13)*acd22(16)
      acd22(45)=acd22(11)*acd22(14)
      acd22(44)=acd22(44)+acd22(45)
      acd22(45)=acd22(44)*acd22(15)
      acd22(46)=acd22(5)*acd22(8)
      acd22(47)=acd22(3)*acd22(7)
      acd22(46)=acd22(46)+acd22(47)
      acd22(47)=acd22(46)*acd22(6)
      acd22(42)=acd22(42)+acd22(45)+acd22(47)+acd22(43)-acd22(17)
      acd22(43)=acd22(23)*acd22(44)
      acd22(44)=acd22(18)*acd22(46)
      acd22(45)=acd22(13)*acd22(25)
      acd22(46)=acd22(11)*acd22(24)
      acd22(45)=acd22(27)+acd22(45)+acd22(46)
      acd22(46)=acd22(9)*acd22(45)
      acd22(47)=acd22(3)*acd22(19)
      acd22(48)=acd22(5)*acd22(20)
      acd22(47)=acd22(47)+acd22(21)+acd22(48)
      acd22(47)=acd22(1)*acd22(47)
      acd22(48)=acd22(36)*acd22(35)
      acd22(49)=acd22(34)*acd22(33)
      acd22(50)=acd22(32)*acd22(31)
      acd22(51)=acd22(17)*acd22(22)
      acd22(52)=acd22(16)*acd22(29)
      acd22(53)=acd22(14)*acd22(28)
      acd22(54)=acd22(8)*acd22(30)
      acd22(55)=acd22(7)*acd22(26)
      acd22(43)=acd22(47)+acd22(46)+acd22(55)+acd22(54)+acd22(53)+acd22(52)-2.0&
      &_ki*acd22(51)+acd22(50)+acd22(48)+acd22(49)+acd22(44)+acd22(43)
      acd22(44)=ninjaP*acd22(42)
      acd22(45)=acd22(23)*acd22(45)
      acd22(46)=acd22(5)*acd22(18)
      acd22(46)=acd22(46)+acd22(30)
      acd22(46)=acd22(20)*acd22(46)
      acd22(47)=acd22(3)*acd22(18)
      acd22(47)=acd22(47)+acd22(26)
      acd22(47)=acd22(19)*acd22(47)
      acd22(48)=acd22(36)*acd22(40)
      acd22(49)=acd22(34)*acd22(39)
      acd22(50)=acd22(32)*acd22(38)
      acd22(51)=acd22(25)*acd22(29)
      acd22(52)=acd22(24)*acd22(28)
      acd22(53)=-acd22(17)*acd22(37)
      acd22(54)=acd22(18)*acd22(21)
      acd22(44)=acd22(44)+acd22(54)+acd22(53)+acd22(52)+acd22(51)+acd22(50)+acd&
      &22(49)+acd22(41)+acd22(48)+acd22(45)+acd22(47)+acd22(46)
      brack(ninjaidxt1mu0)=acd22(43)
      brack(ninjaidxt1mu2)=0.0_ki
      brack(ninjaidxt0mu0)=acd22(44)
      brack(ninjaidxt0mu2)=acd22(42)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d22h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd22h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k5
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d22h0l131
