module     p2_gg_httbar_d162h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d162h0l1.f90
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
      use p2_gg_httbar_abbrevd162h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc162(30)
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1e2
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      acc162(1)=abb162(12)
      acc162(2)=abb162(13)
      acc162(3)=abb162(14)
      acc162(4)=abb162(15)
      acc162(5)=abb162(16)
      acc162(6)=abb162(19)
      acc162(7)=abb162(20)
      acc162(8)=abb162(21)
      acc162(9)=abb162(23)
      acc162(10)=abb162(24)
      acc162(11)=abb162(25)
      acc162(12)=abb162(26)
      acc162(13)=abb162(27)
      acc162(14)=abb162(29)
      acc162(15)=abb162(34)
      acc162(16)=abb162(35)
      acc162(17)=abb162(40)
      acc162(18)=abb162(44)
      acc162(19)=abb162(51)
      acc162(20)=abb162(70)
      acc162(21)=acc162(6)*Qspvak2e1
      acc162(22)=acc162(12)*Qspvae2e1
      acc162(23)=acc162(14)*Qspval5e1
      acc162(24)=acc162(15)*Qspval4e1
      acc162(25)=acc162(18)*Qspvak1e1
      acc162(21)=acc162(25)+acc162(24)+acc162(23)+acc162(22)+acc162(21)+acc162(&
      &2)
      acc162(21)=Qspvae1k2*acc162(21)
      acc162(22)=acc162(1)*Qspvae1l5
      acc162(23)=acc162(3)*Qspvae1k1
      acc162(24)=-acc162(7)*Qspvae1l4
      acc162(25)=-acc162(17)*Qspvae1e2
      acc162(22)=acc162(25)+acc162(24)+acc162(4)+acc162(23)+acc162(22)
      acc162(22)=Qspval5e1*acc162(22)
      acc162(23)=acc162(5)*Qspvak1e1
      acc162(24)=acc162(8)*Qspvae1e2
      acc162(25)=acc162(9)*Qspvak2e1
      acc162(26)=acc162(10)*Qspvae1k1
      acc162(27)=acc162(13)*Qspvae1l4
      acc162(28)=acc162(16)*Qspval4e1
      acc162(29)=acc162(19)*Qspvae2e1
      acc162(30)=acc162(20)*Qspvae1l5
      brack=acc162(11)+acc162(21)+acc162(22)+acc162(23)+acc162(24)+acc162(25)+a&
      &cc162(26)+acc162(27)+acc162(28)+acc162(29)+acc162(30)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d162h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd162h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d162
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d162 = 0.0_ki
      d162 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d162, ki), aimag(d162), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d162h0l1
