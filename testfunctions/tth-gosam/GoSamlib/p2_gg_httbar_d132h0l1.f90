module     p2_gg_httbar_d132h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d132h0l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd132h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc132(30)
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2e1
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      acc132(1)=abb132(12)
      acc132(2)=abb132(13)
      acc132(3)=abb132(14)
      acc132(4)=abb132(15)
      acc132(5)=abb132(16)
      acc132(6)=abb132(17)
      acc132(7)=abb132(18)
      acc132(8)=abb132(19)
      acc132(9)=abb132(21)
      acc132(10)=abb132(22)
      acc132(11)=abb132(23)
      acc132(12)=abb132(24)
      acc132(13)=abb132(27)
      acc132(14)=abb132(28)
      acc132(15)=abb132(34)
      acc132(16)=abb132(36)
      acc132(17)=abb132(37)
      acc132(18)=abb132(39)
      acc132(19)=abb132(43)
      acc132(20)=abb132(88)
      acc132(21)=acc132(2)*Qspvak2e2
      acc132(22)=acc132(5)*Qspval5e2
      acc132(23)=acc132(6)*Qspval4e2
      acc132(24)=acc132(8)*Qspvak1e2
      acc132(25)=acc132(13)*Qspvae1e2
      acc132(21)=acc132(25)+acc132(24)+acc132(23)+acc132(22)+acc132(3)+acc132(2&
      &1)
      acc132(21)=Qspvae2k2*acc132(21)
      acc132(22)=-acc132(7)*Qspvae2l4
      acc132(23)=acc132(10)*Qspvae2k1
      acc132(24)=acc132(12)*Qspvae2l5
      acc132(25)=-acc132(20)*Qspvae2e1
      acc132(22)=acc132(25)+acc132(24)+acc132(23)+acc132(22)+acc132(1)
      acc132(22)=Qspval5e2*acc132(22)
      acc132(23)=acc132(4)*Qspvae2k1
      acc132(24)=acc132(9)*Qspvak2e2
      acc132(25)=acc132(11)*Qspvak1e2
      acc132(26)=acc132(14)*Qspvae2l4
      acc132(27)=acc132(16)*Qspvae2l5
      acc132(28)=acc132(17)*Qspvae2e1
      acc132(29)=acc132(18)*Qspval4e2
      acc132(30)=acc132(19)*Qspvae1e2
      brack=acc132(15)+acc132(21)+acc132(22)+acc132(23)+acc132(24)+acc132(25)+a&
      &cc132(26)+acc132(27)+acc132(28)+acc132(29)+acc132(30)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d132h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd132h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d132
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d132 = 0.0_ki
      d132 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d132, ki), aimag(d132), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d132h0l1
